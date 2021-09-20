# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 15:29:57 2018

@author: jackie
"""

import matplotlib.pyplot as plt
import numpy as np
import cPickle as pkl
import random
import numpy.fft as fft 
from scipy.signal import correlate
from obspy.taup import TauPyModel
from rf import rfstats, RFStream
from obspy.core.inventory import read_inventory, Channel, inventory 
from obspy.clients.fdsn.client import Client
from obspy.core import UTCDateTime, read, Stream
from scipy import signal
from obspy.signal.filter import bandpass, lowpass
import copy 
import os,glob,subprocess,obspy,cPickle
from obspy.core.event import Origin, Event, Catalog
from obspy.geodetics import gps2dist_azimuth, locations2degrees



################################################################################
################################################################################
def cut_events(catalog,stn,net,statdir,workdir,interval=[-60.,540.],
               phase='P',Nmax = 10):
  """
  cut events (P phases) from either local data or by retrieving data from IRIS...careful: channel identifiers (BH,HH) or location codes ('--','00','10' etc) may have to be changed
  input:
  catalog - obspy catalog 
  stn - obspy station object
  station - station name (string)
  netwrk - network name (string)
  interval - time window around theo. P pick to cut out (values in seconds)
  """

  os.chdir(statdir)
  sta_lat  = stn.latitude
  sta_lon  = stn.longitude 
  sta_elev = stn.elevation
  station = stn.code
  netwrk = net.code
  tmp = np.zeros(catalog.count())
  num = 0 ## Counter for No. retained events
  iphase = 0
  ifol = 0
  idown = 0

  fevnt4 = open(statdir + 'debg_eventnm_catlg4.txt','w')
  catalog2 = Catalog()
  for i,evnt in enumerate(catalog):
    orgn = evnt.preferred_origin()
    print(i)
    tmp[i] = '%4i%02i%02i%02i%02i%02i' % (orgn.time.year,orgn.time.month,
             orgn.time.day,orgn.time.hour,orgn.time.minute,orgn.time.second)
    fevnt4.write(repr(i)+'  '+repr(tmp[i]) + '\n')
    magn = evnt.preferred_magnitude()
    if num >= Nmax:
      break
    if i%100 == 0:
      print 'Downloading event ',i,' of magnitude = ',magn.mag,' of ',len(catalog),'events.'
    distm,az,baz = gps2dist_azimuth(orgn.latitude,orgn.longitude,sta_lat,sta_lon)
    delta = locations2degrees(orgn.latitude,orgn.longitude,sta_lat,sta_lon)
    model = TauPyModel(model="ak135")
    depth = orgn.depth
    if depth > 1000.:
      depth = depth/1000.
    ttdic = model.get_travel_times(depth,delta,phase_list=[phase])

    try:
      ttime = ttdic[0]
      ttime=ttime.time
    except IndexError: #phase does not exist
      iphase += 1
      fevnt4.write(' Phase error ' + repr(iphase) +'\n')
      continue
    tof = ttdic[0].takeoff_angle
    phaseflag = True
   
    arrival = orgn.time + ttime
    t1 = arrival + interval[0]
    t2 = arrival + interval[1]

    datestrg_folder = '%4i%02i%02i_%02i%02i%02i%06i' % (orgn.time.year,
                      orgn.time.month,orgn.time.day,orgn.time.hour,
                      orgn.time.minute,orgn.time.second,orgn.time.microsecond)
   
    #create event folder
    try:
      os.mkdir(statdir + 'event_'+datestrg_folder)
    except: #already exists
      ifol += 1
      fevnt4.write(' Folder exists already! ' + repr(ifol)+'\n')
      pass
    os.chdir(statdir + 'event_'+datestrg_folder)
    datestrg_file = '%4i.%03i.%02i.%02i.%02i' % (orgn.time.year,orgn.time.julday,
                      orgn.time.hour,orgn.time.minute,orgn.time.second)
   
    from obspy.clients.fdsn import Client
    client = Client("IRIS")
    #download data from IRIS  
    try:
      tr_z = client.get_waveforms(netwrk, station,'*', 'BHZ', t1, t2)
      tr_e = client.get_waveforms(netwrk, station,'*', 'BHE', t1, t2)
      tr_n = client.get_waveforms(netwrk, station,'*', 'BHN', t1, t2)
 
      tr_z.merge(fill_value='interpolate')
      tr_n.merge(fill_value='interpolate')
      tr_e.merge(fill_value='interpolate')
 
      tr_z.write(datestrg_file+'.'+station+'.'+phase+'.BHZ.SAC','SAC')
      tr_n.write(datestrg_file+'.'+station+'.'+phase+'.BHN.SAC','SAC')
      tr_e.write(datestrg_file+'.'+station+'.'+phase+'.BHE.SAC','SAC')
      print('Kept ', str(i)) 
    except Exception:
      #print('Reject ', str(i))  
      os.chdir(statdir)
      os.system("rm -rf event_"+datestrg_folder)
      idown += 1
      fevnt4.write(' Deleting ' + repr(idown)+'\n')
      continue
 
    #set SAC headers!!
    from obspy.io.sac.sactrace import SACTrace
    for comp in ['E','N','Z']:
      hd = SACTrace.read(datestrg_file+'.'+station+'.'+phase+'.BH'+comp+'.SAC',headonly=True)
      hd.mag = magn.mag
      hd.baz = baz
      hd.stla = sta_lat
      hd.stlo = sta_lon
      hd.evlo = orgn.longitude
      hd.evla = orgn.latitude
      hd.evdp = depth
      hd.gcarc = delta
      hd.a = arrival  # P arrival marker (arrival time in s)
      hd.write(datestrg_file+'.'+station+'.'+phase+'.BH'+comp+'.SAC',headonly=True)
    os.chdir(statdir)
    num += 1
    fevnt4.write(' Success ' + repr(i) +'  '+ repr(num)+'  '+ repr(iphase)+'  '+ repr(ifol)+'  '+ repr(idown) + '\n')
    fevnt4.write('\n'+'\n')
    catalog2.append(catalog[i])
  os.chdir(workdir)
  return catalog2

################################################################################
################################################################################
def prepare_traces(statdir,catlg,fmin=0.1,fmax=4,cutaway=[30,120],rotate='ZRT',
                   phase='P', displ=False,Nmax = 10, snrlen = 20, snrzcut = 5, snrnecut = 1, parr = 70):
  """a =	F	First arrival time (seconds relative to reference time.)
  cut (shorten) and prepare traces for CC analysis and stacking
  input:
  statdir - path of the data to be used (folders event*)
  fmin - minimum frequency for bandpass filter
  fmax - maximum frequency for bandpass filter
  cutaway - time windows (seconds) to cut off from raw trace (at beginning and end)
  rotate - RTZ or LQT
  """

  ferr = open(statdir + 'prepare_traces_errors.txt','w')
  fevnt = open(statdir + 'debg_eventnm_catlg.txt','w')
  fevnt2 = open(statdir + 'debg_eventnm_catlg2.txt','w')
  fevnt3 = open(statdir + 'debg_eventnm_catlg3.txt','w')

  #create big dictionary with traces
  infolders = glob.glob(statdir+'/event*')
  num = 0
  loopidx = 0
    
  print('No. events on disk:',len(infolders) )
  print('No. events in Catalog:',catlg.count() )
  print statdir

  eventdict = {}
  eventdict_unrot = {}

  for eventnm in sorted(infolders,reverse=True):
#    print eventnm
    evtmp = catlg[loopidx]
    orgn = evtmp.preferred_origin()
      
    if num >= Nmax:
      break
    stream = read(eventnm+'/[1,2]*.BH[E,N,Z].SAC')
    fevnt.write(repr(num)+ '  '+repr(loopidx)+ '  ' + eventnm[-21:] + '  ' + repr(orgn.time) + '\n')
    fevnt3.write(repr(num)+ '  '+repr(loopidx)+ '  ' + eventnm[-15:-7] + eventnm[-6:] + '  ' + repr(orgn.time) + '\n')
    loopidx += 1
    #check if we really have 3 components
    if not len(stream) == 3:
      ferr.write('Wrong number of components: ' + eventnm + '\n')
      continue

    #check that this length is not zero
    if len(stream[0]) == 0 or len(stream[1]) == 0 or len(stream[2]) == 0:
      ferr.write("Zero-length stream encountered...skipping: " + eventnm + '\n')
      continue

    eventdict[num] = {}
    eventdict_unrot[num] = {}
    
    model = TauPyModel(model="ak135")
    
    delta = stream[0].stats.sac.delta
            
    try:
      tt = model.get_travel_times(stream[0].stats.sac.evdp, delta)
      ##slow = (tt[0].ray_param)* math.pi /180/111.195 #slowness convert from s/rad to s/km
      ##slow = (tt[0].ray_param)
      ttime = tt[0].time
      slow = (tt[0].ray_param_sec_degree)/111.195
      
    except:
    #  print('An error occurred')
      continue
    arrival = orgn.time + ttime
    stream.detrend(type='demean')
    stream.detrend(type='linear')
    
    #CHECK SNR HERE AND GET RID OF LOW ENTRIES
    sparr = np.rint(parr/delta).astype(int)

    sgn = stream[2].data[sparr:sparr+snrlen]
    nse = stream[2].data[0:2*snrlen]
    Pow_s = np.sum(sgn**2)/snrlen
    Pow_n = np.sum(nse**2)/(2*snrlen)
    SNR_Z = 10.*np.log10(Pow_s/Pow_n)
    
    #SNR for E
    sgn_e = stream[0].data[sparr:sparr+snrlen]
    nse_e = stream[0].data[0:2*snrlen]
    #Pow_se = np.sum(sgn_e**2)/snrlen
    #Pow_ne = np.sum(nse_e**2)/(2*snrlen)

    #SNR for N
    sgn_n = stream[1].data[sparr:sparr+snrlen]
    nse_n = stream[1].data[0:2*snrlen]
    #Pow_sn = np.sum(sgn_n**2)/snrlen
    #Pow_nn = np.sum(nse_n**2)/(2*snrlen)

    #Find the SNR for N/E combo
    #SNR_NE = 10.*np.log10((np.sqrt(Pow_se) + np.sqrt(Pow_sn))/(np.sqrt(Pow_ne)+ np.sqrt(Pow_nn)))
    
    #print(num, SNR_Z, SNR_NE)
    if SNR_Z < snrzcut: # or SNR_NE < snrnecut:
      ferr.write("SNR too high! Skipping: " + eventnm + '\n')
      continue

          
    eventdict_unrot[num]['event'] = evtmp
    eventdict_unrot[num]['evstr'] = eventnm
    eventdict_unrot[num]['stream'] = stream
    eventdict_unrot[num]['baz'] = stream[0].stats.sac.baz
    eventdict_unrot[num]['magnitude'] = stream[0].stats.sac.mag
    eventdict_unrot[num]['depth'] = stream[0].stats.sac.evdp
    eventdict_unrot[num]['parr'] = arrival
    delta = stream[0].stats.sac.gcarc
    eventdict_unrot[num]['distance'] = delta

    if rotate == 'ZRT':
       #check that they have same length
      if not len(stream.select(component='E')[0]) == len(stream.select(component='N')[0]): #necessary for rotation
        ferr.write("Components have different lengths! Event skipped " + eventnm + '\n')
        continue
      if not stream[0].stats.starttime == stream[1].stats.starttime: #necessary for rotation
        ferr.write("Components have different starttime! Event skipped " + eventnm + '\n')
        continue
      if not stream[0].stats.starttime == stream[2].stats.starttime: #necessary for rotation
        ferr.write("Components have different starttime! Event skipped " + eventnm + '\n')
        continue
      if not stream[1].stats.starttime == stream[2].stats.starttime: #necessary for rotation
        ferr.write("Components have different starttime! Event skipped " + eventnm + '\n')
        continue
      #plt.figure()
      #plt.plot(stream[1].data)
      stream.rotate(method='NE->RT',back_azimuth=stream[0].stats.sac.baz)
      
      #plt.figure()
      #plt.plot(stream[1].data)
      
    elif rotate == 'LQT':
      print(rotate)
      if not len(stream.select(component='E')[0]) == len(stream.select(component='N')[0]) == len(stream.select(component='Z')[0]): #necessary for rotation
        ferr.write("Components have different lengths! Event skipped" + eventnm + '\n')
        continue

      #how to get incidence angle? Currently from 1D model -- better to minimize energy??
      model = TauPyModel(model='ak135')
      arr2 = model.get_travel_times(stream[0].stats.sac.evdp,stream[0].stats.sac.gcarc,phase_list=['P'])
      inc = arr2[0].incident_angle
      stream.rotate(method='ZNE->LQT',back_azimuth=stream[0].stats.sac.baz,inclination=inc)
    
    elif rotate == 'PVH':
      print(rotate)
       #Rotate it to RTZ so we can apply the rotation matrix
      if not len(stream.select(component='E')[0]) == len(stream.select(component='N')[0]): #necessary for rotation
        ferr.write("Components have different lengths! Event skipped " + eventnm + '\n')
        continue
      if not stream[0].stats.starttime == stream[1].stats.starttime: #necessary for rotation
        ferr.write("Components have different starttime! Event skipped " + eventnm + '\n')
        continue
      if not stream[0].stats.starttime == stream[2].stats.starttime: #necessary for rotation
        ferr.write("Components have different starttime! Event skipped " + eventnm + '\n')
        continue
      if not stream[1].stats.starttime == stream[2].stats.starttime: #necessary for rotation
        ferr.write("Components have different starttime! Event skipped " + eventnm + '\n')
        continue
      stream.rotate(method='NE->RT',back_azimuth=stream[0].stats.sac.baz)
      model = TauPyModel(model='ak135')
      if stream[0].stats.sac.gcarc == -12345.:
          distm,az,baz = gps2dist_azimuth(isc[i]['event_lat'],isc[i]['event_lon'],sta_lat,sta_lon)
          delta = distm / (1000.*111.195)
      else:
          delta = stream[0].stats.sac.gcarc 
      tt = model.get_travel_times(stream[0].stats.sac.evdp, delta)
      ##slow = (tt[0].ray_param)* math.pi /180/111.195 #slowness convert from s/rad to s/km
      ##slow = (tt[0].ray_param)
      slow = (tt[0].ray_param_sec_degree)/111.195
      #approximate Vp (alpha) and Vs (beta)    
      alph = 6.
      beta = alph/1.75
      #calculate qalph and qbeta  
      qalph = np.sqrt(alph ** (-2)-slow**2)
      qbeta = np.sqrt(beta**(-2)-slow**2) 

      #calculate values for rotation matrix   
      VPZ = (1.-2.*beta**2*slow**2)/(2.*alph*qalph)
      VPR = (slow*beta**2)/alph 
      VSZ = -slow * beta
      VSR = (1.-2.*beta**2*slow**2)/(2.*beta*qbeta)
      VHT = 0.5 
      #rotation matrix defined by Bostock, 1998  
      rotmat_lqt = np.array([[VPZ, VPR, 0.],[VSZ, VSR, 0.],[0., 0., VHT]])
      print(eventnm)
      #current values of  uz, ur, ut
      #multiply and assign to P, SV, SH
      (pst, svst, shst)= np.matmul(rotmat_lqt.astype(float),np.array([stream[2].data.astype(float), stream[1].data.astype(float), stream[0].data.astype(float)]))
      stream[2].data = pst
      stream[1].data = svst
      stream[0].data = shst
        
    #stream.filter('bandpass',freqmin=fmin,freqmax=fmax,corners=4,zerophase=False)
    #stream.filter('lowpass', freq=fmax, corners=4, zerophase = False)
    #stream.filter('lowpass', freq=fmax, corners=4, zerophase = True)
    
    
    try:
      stream_new = stream.slice(stream[0].stats.starttime + cutaway[0], stream[0].stats.starttime + cutaway[1])
    except ValueError:
      continue

    #print(num, len(stream_new[0].data) - (cutaway[1] - cutaway[0])*stream_new[0].stats.sampling_rate)

    #check if traces are available
    if not len(stream_new) == 3:
      ferr.write("Error: Stream has only one trace! Skipping..." + eventnm + '\n')
      continue
    #check whether all traces have correct length
    #allow for two samples difference
    if not abs(len(stream_new[0].data) - (cutaway[1] - cutaway[0])*stream_new[0].stats.sampling_rate) <= 2:
      ferr.write("Error: cut trace does not have expected length! Skipping..." + eventnm + '\n')
      continue
    if not abs(len(stream_new[1].data) - (cutaway[1] - cutaway[0])*stream_new[0].stats.sampling_rate) <= 2:
      ferr.write("Error: cut trace does not have expected length! Skipping..." + eventnm + '\n')
      continue
    if not abs(len(stream_new[2].data) - (cutaway[1] - cutaway[0])*stream_new[0].stats.sampling_rate) <= 2:
      ferr.write("Error: cut trace does not have expected length! Skipping..." + eventnm + '\n')
      continue

    #check for flat components
    if abs(stream_new[0].data.max() - stream_new[0].data.min()) < 5:
      ferr.write("Flat component! Rejected.")
      continue
    if abs(stream_new[1].data.max() - stream_new[0].data.min()) < 5:
      ferr.write("Flat component! Rejected.")
      continue
    if abs(stream_new[2].data.max() - stream_new[0].data.min()) < 5:
      ferr.write("Flat component! Rejected.")
      continue


    #demean and detrend again (in case of highly volatile traces)
    stream_new.detrend(type='demean')
    stream_new.detrend(type='linear')

    #downsample to 10 Hz
    factor = int(stream_new[0].stats.sampling_rate) / 10
    stream_new.decimate(factor)

#    if paz:
#      stream_new.simulate(paz_remove=pazdict)
    if displ:
      stream_new.integrate()
    try:
        eventdict[num]['event'] = evtmp
        eventdict[num]['evstr'] = eventnm
        eventdict[num]['stream'] = stream_new
        eventdict[num]['baz'] = stream[0].stats.sac.baz
        eventdict[num]['baz2']  = gps2dist_azimuth(orgn['latitude'],orgn['longitude'],stream[0].stats.sac.stla,
                                    stream[0].stats.sac.stlo)
        eventdict[num]['magnitude'] = stream[0].stats.sac.mag
        eventdict[num]['depth'] = stream[0].stats.sac.evdp
        eventdict[num]['evlat'] = stream[0].stats.sac.evla 
        eventdict[num]['evlon'] = stream[0].stats.sac.evlo
        eventdict[num]['parr'] = arrival
        eventdict[num]['rot'] = rotate
        if stream[0].stats.sac.gcarc == -12345.:
          distm,az,baz = gps2dist_azimuth(isc[i]['event_lat'],isc[i]['event_lon'],sta_lat,sta_lon)
          delta = distm / (1000.*111.195)
        else:
          delta = stream[0].stats.sac.gcarc
       
        eventdict[num]['distance'] = delta
         
        model = TauPyModel(model='ak135')
           
        try:
          tt = model.get_travel_times(stream[0].stats.sac.evdp, delta)
          ##slow = (tt[0].ray_param)* math.pi /180/111.195 #slowness convert from s/rad to s/km
          ##slow = (tt[0].ray_param)
          slow = (tt[0].ray_param_sec_degree)/111.195
          
        except:
        #  print('An error occurred')
          continue
        eventdict[num]['distance'] = delta
        eventdict[num]['slowness'] = slow
        fevnt2.write(repr(num)+ '  '+repr(loopidx-1)+ '  ' + eventnm[-21:] + '  ' + repr(orgn.time) + '\n')
        num += 1
    except: 
        continue
    if (loopidx-1)%100 == 0:
      print'Processed ',loopidx-1,'evets; Retained ',num,' events. Currently processing: ',eventnm
  ferr.close()

  return eventdict, eventdict_unrot


################################################################################
################################################################################
def rot_rtz2psv(evdict, beta=3.45, alph=6):
    """
    Rotate RZT to PSV
    Requires: 
    evdict = event dictionary 
    beta = s wave vel
    alph = p wave vel
    outputs: 
    evdict = event dictionary with P, SV, SH 
    """
    #initiazlie for rotation 
    hldr = np.zeros(3)

    NEV = len(evdict)
    slow = []
    magn = []
    baz= []
        #loop through the time series 
    for iev in range(0, NEV):

      evtst = evdict[iev]['stream']
      event = evdict[iev]

      slow= event.get('slowness') 
      magn.append(event.get('magnitude'))
      baz.append(event.get('baz'))
      
      qalph = np.sqrt(alph ** (-2)-slow**2)
      qbeta = np.sqrt(beta**(-2)-slow**2) 
      
      #rotation components
      VPZ = -(1-2*beta*beta*slow*slow)/(2*alph*qalph)
      VPR = slow*beta*beta/alph 
      VSZ = slow*beta
      VSR = (1-2*beta*beta*slow*slow)/(2*beta*qbeta)
      VHT = 0.5   
    
      #rotation matrix  
      rotmat_pvh = np.array([[VPZ, -VPR, 0.],[VSZ, -VSR, 0.],[0., 0., VHT]])
      
      #apply rotation matrix
      hldr= np.matmul(rotmat_pvh,[evtst[2].data, evtst[1].data, evtst[0].data])
      
      
      evdict[iev]['stream'][2].data = hldr[0]
      evdict[iev]['stream'][1].data = hldr[1]
      evdict[iev]['stream'][0].data = evtst[0].data*VHT
      evdict[iev]['rot'] = 'PVH'
      
    return evdict
################################################################################
################################################################################  

def event_select(eventdict,fmin,fmax,magrange=[6.0,9.0],bazrange=[0.,90.],
                 distrange=[30.,95.],deprange=[0.,700.],slowrange=[0.05,0.07],islow='False'):
  """
  select events by defining ranges of magnitude, baz and slowness
  copied from direct_stack by J.Dettmer
  
  """
  num_new = 0
  dict_new = {}
   
  if islow == 'False':
    for j in eventdict.keys():
        try: 
            print j
            if eventdict[j]['magnitude'] >= magrange[0] and eventdict[j]['magnitude'] <= magrange[1]:
             if eventdict[j]['baz'] >= bazrange[0] and eventdict[j]['baz'] <= bazrange[1]:
              if eventdict[j]['distance'] >= distrange[0] and eventdict[j]['distance'] <= distrange[1]:
                if eventdict[j]['depth'] >= deprange[0] and eventdict[j]['depth'] <= deprange[1]:
                      dict_new[num_new] = copy.deepcopy(eventdict[j])
                      num_new += 1
        except: 
            continue 
  else:
    for j in eventdict.keys():
        try: 
          print(j)
          if eventdict[j]['magnitude'] >= magrange[0] and eventdict[j]['magnitude'] <= magrange[1]:
            if eventdict[j]['baz'] >= bazrange[0] and eventdict[j]['baz'] <= bazrange[1]:
              if eventdict[j]['slowness'] >= slowrange[0] and eventdict[j]['slowness'] <= slowrange[1]:
                    dict_new[num_new] = copy.deepcopy(eventdict[j])
                    num_new += 1
        except:
            continue

  if dict_new == {}:
    print "Selection criteria leave no valid events...need to be changed!!"
  #dump choices
  tempfile = open('choices.tmp','w')
  tempfile.write('Mag low, Mag high, BAZ low, BAZ high, Dist low, Dist high, fmin, fmax')
  tempfile.write('%4.2f %4.2f %6.2f %6.2f %7.3f %7.3f %5.2f %5.2f' % (magrange[0],magrange[1],
                 bazrange[0],bazrange[1],distrange[0],distrange[1], fmin, fmax))
  tempfile.close()

  return dict_new 
        
################################################################################
################################################################################

def process_rf(eventdict,stn_coords,Nmax=10):
  """
  Process event dictionary into RF dictionary
  """
  loopidx = 0
  rfdict = {}
  
  for iev in eventdict:
    #print(iev)
    if loopidx >= Nmax:
      break
    rfevent = copy.deepcopy(eventdict[iev])
    #st = eventdict[iev].get('stream')
    st = rfevent.get('stream')
    #ev = eventdict[iev].get('event')
    ev = rfevent.get('event')
    rfst = RFStream(st)
    stats = rfstats(station=stn_coords, event=ev, phase='P', dist_range=(30,95))
    for tr in rfst:
      tr.stats.update(stats)
    rfst.filter('bandpass', freqmin=0.05, freqmax=1.)
    rfst.rf(method='P',deconvolve='freq', waterlevel=0.005, gauss=1.5)
#    rfst.rf(method='P',deconvolve='time')
    rfevent['stream'] = rfst
    rfdict[iev] = rfevent
    rfdict[iev]['rf'] = 'WTR'
    loopidx += 1

  return rfdict        
        
        
################################################################################
################################################################################
def snr_calc(evdict, rflen= 50, parr=70):
    """"
    Calculate SNR for unrotated traces
    requires: 
    evdict
    rflen = seconds of rf length 
    parr = seconds expected p arrival 
    outputs: 
    SNR_NE = snr on the ne component
    SNR_Z = snr on the z component 
    
    
    NOT CURRENTLY USED IN RF_PROCESSING
   """" "     
    #Stream and smapline rate for lenghts 
    st = evdict[0]['stream']
    delta = st[0].stats.sampling_rate
    srflen = int(rflen*delta) #How many samples for Audet's RF/SNR? 
    sparr = int(parr*delta) #expected P arr for rfdict
    sparr_unrot = int(100*delta) #Which sample expected P-arrival for unrotated? 
       
    #initiazlie 
    SNR_N = np.zeros(len(evdict))
    SNR_E = np.zeros(len(evdict))
    SNR_Z = np.zeros(len(evdict))
    SNR_NE = np.zeros(len(evdict))
    
    #Loop through and calculate the SNR on N,E and Z componenets 
    for iev in range(len(evdict)): 
        st = evdict[iev]['stream'] # stream 
        #SNR for Z 
        sgn = st[2].data[sparr_unrot:sparr+srflen]
        nse = st[2].data[0:srflen]
        Pow_s = np.sum(sgn**2/srflen)
        Pow_n = np.sum(nse**2/srflen)
        SNR_Z[iev] = 10.*np.log10(Pow_s/Pow_n)
        #SNR for E
        sgn_e = st[0].data[sparr_unrot:sparr+srflen]
        nse_e = st[0].data[0:srflen]
        Pow_s = np.sum(sgn_e**2/srflen)
        Pow_n = np.sum(nse_e**2/srflen)
        SNR_E[iev] = 10.*np.log10(Pow_s/Pow_n)
        #SNR for N
        sgn_n = st[1].data[sparr_unrot:sparr+srflen]
        nse_n = st[1].data[0:srflen]
        Pow_s = np.sum(sgn_n**2)/srflen
        Pow_n = np.sum(nse_n**2)/srflen
        SNR_N[iev] = 10.*np.log10(Pow_s/Pow_n)
        #Find the SNR for N/E combo
        SNR_NE[iev] = 10.*np.log10(((np.sqrt(np.sum(sgn_e**2)/srflen) 
            + np.sqrt(np.sum(sgn_n**2)/srflen))/(np.sqrt(np.sum(nse_e**2)/srflen) 
            + np.sqrt(np.sum(nse_n**2)/srflen)))**2)
        
    return SNR_NE, SNR_Z        
        

        
#############################################################################
###############################################################################
def wiener_rf(evdict, rflen= 50, parr=70, tapelen=50, flo=0.1, fhi=1.0):     
    """
    Wiener filter RF calculation 
    requires 
    evdict
    rflen = length of rf we want
    parr = expected parrival 
    tapelen = taper length
    flo = low filter for taper
    fhi = high filter for taper 
    """
    #Define necessary/arbitrary variables  
    delta = evdict[0]['stream'][0].stats.sampling_rate
    tapelen = 50 
    srflen = int(rflen*delta) #samples for Noise/RF invento
    sparr = int(parr*delta) # Start of window just before expected sample arrival of P-wave  
    
    Sp =   np.zeros(len(evdict[0]['stream'][2]), dtype=np.complex_)      
    Sv =   np.zeros(len(evdict[1]['stream'][2]), dtype=np.complex_) 
    #initiazlie 
    rfP = np.zeros([len(evdict[0]['stream'][2]),srflen])
    rfV = np.zeros([len(evdict[0]['stream'][1]),srflen])
    rfH = np.zeros([len(evdict[0]['stream'][0]),srflen])
    
    #apply taper 
    t2 = (np.cos(np.arange(0.,np.pi,np.pi/float(tapelen)))+1.)/2.
    t1 = (np.cos(np.arange(np.pi,2.*np.pi,np.pi/float(tapelen)))+1.)/2.
    
    taper = np.ones(srflen)
    taper[0:tapelen] = t1
    taper[-tapelen:] = t2
    
    
    rfdict= copy.deepcopy(evdict)         
    for iev in range(len(rfdict)):    
        
        #apply butterworth filter 
        nP= bandpass(rfdict[iev]['stream'][2][0:srflen] , flo, fhi, delta,corners = 4, zerophase = True)
        P = bandpass(rfdict[iev]['stream'][2][sparr-tapelen:sparr+srflen-tapelen], flo, fhi , delta,corners = 4, zerophase = True)
        nSV= bandpass(rfdict[iev]['stream'][1][0:srflen], flo, fhi , delta,corners = 4, zerophase = True)
        SV = bandpass(rfdict[iev]['stream'][1][sparr-tapelen:sparr+srflen-tapelen], flo, fhi , delta,corners = 4, zerophase = True)
        nSH= bandpass(rfdict[iev]['stream'][0][0:srflen], flo, fhi , delta,corners = 4, zerophase = True)
        SH = bandpass(rfdict[iev]['stream'][0][sparr-tapelen:sparr+srflen-tapelen], flo, fhi , delta,corners = 4, zerophase = True)
    
        #Src = bandpass(rfdict[iev]['stream'][2][sparr-20-tapelen:sparr+srflen-20-tapelen], flo, fhi , delta,corners = 4, zerophase = True)
        Src = bandpass(rfdict[iev]['stream'][2][sparr-20:sparr+srflen-20], flo, fhi , delta,corners = 4, zerophase = True)
        
        #Apply Taper 
        nP = nP * taper
        P = P * taper 
        nSV = nSV * taper 
        SV = SV * taper 
        nSH = nSH * taper 
        SH = SH * taper         
        
        #apply butterworth filter 
        nP= bandpass(nP, flo, fhi, delta,corners = 4, zerophase = True)
        P = bandpass(P, flo, fhi , delta,corners = 4, zerophase = True)
        nSV= bandpass(nSV, flo, fhi , delta,corners = 4, zerophase = True)
        SV = bandpass(SV, flo, fhi , delta,corners = 4, zerophase = True)
        nSH= bandpass(nSH, flo, fhi , delta,corners = 4, zerophase = True)
        SH = bandpass(SH, flo, fhi , delta,corners = 4, zerophase = True)

        Src = bandpass(Src, flo, fhi , delta,corners = 4, zerophase = True)
        twin = len(P)/delta
       # twin = 10
        window = np.zeros(len(P))
        
        #_taper function for src
        nt = int(twin*delta)
        ns = int(2*delta)
        tap = np.ones(nt)
        win = np.hanning(2*ns)
        tap[0:ns] = win[0:ns]
        tap[nt-ns:nt] = win[ns:2*ns]
        window[0:int(twin*delta)] = tap
        Src *= window


        #Apply the taper   
        nP = nP * taper
        P = P * taper 
        nSV = nSV * taper 
        SV = SV * taper 
        nSH = nSH * taper 
        SH = SH * taper 
    
        # fft 
        Fnp = fft.fft(nP)
        Fp = fft.fft(P)
        Fv = fft.fft(SV)
        Fnv =fft.fft(nSV)
        Fh =fft.fft(SH)
        Fs = fft.fft(Src)

       # Auto and cross spectra
        Sp = Fp*np.conjugate(Fs)
        Sv = Fv*np.conjugate(Fs)
        Sh = Fh*np.conjugate(Fs)
        Ss = Fs*np.conjugate(Fs)
        Snp = Fnp*np.conjugate(Fnp)
        Snv = Fnv*np.conjugate(Fnv)
        Snpv = Fnv*np.conjugate(Fnp)
                
        #Denominator for ifft        
        Sdenom = 0.25*(Snp+Snv)+0.5*(Snpv)
    
        rfP[iev] = fft.ifft(Sp/(Ss+Sdenom))
        print(np.amax(rfP[iev]))
        rfV[iev] = fft.ifft(Sv/(Ss+Sdenom))/np.amax(rfP[iev])
        rfH[iev]= fft.ifft(Sh/(Ss+Sdenom))/np.amax(rfP[iev])        
        
        #update RF dict, mark which kind of rf we used
        rfdict[iev]['stream'][2].data = rfP[iev]
        rfdict[iev]['stream'][1].data = rfV[iev]
        rfdict[iev]['stream'][0].data = rfH[iev]
        rfdict[iev]['rf'] = 'WNR'
    
    return rfdict            
 #############################################################################
###############################################################################
def water_rf(evdict,tapelen=50, flo=0.1, fhi=1.0, c0 =0.005, ag = 5, t0 = 5):     
    """
    Waterlevel RF calculation 
    requires 
    evdict

    parr = expected parrival 
    tapelen = taper length
    flo = low filter for taper
    fhi = high filter for taper     
    c0 = 0.005 #waterlevel 
    ag = 5 #Gaussian filter width 
       """ 
    #Define necessary/arbitrary variables  
    delta = evdict[0]['stream'][0].stats.sampling_rate #sampling rate
    ns = len(evdict[0]['stream'][0])
    phi = np.zeros(ns)
    gauf = np.zeros(ns)
    cs = np.zeros(ns, dtype=np.complex_)
    
    #Create taper 
    t2 = (np.cos(np.arange(0.,np.pi,np.pi/float(tapelen)))+1.)/2.
    t1 = (np.cos(np.arange(np.pi,2.*np.pi,np.pi/float(tapelen)))+1.)/2.
    taper = np.ones(len(evdict[0]['stream'][0].data))
    taper[0:tapelen] = t1
    taper[-tapelen:] = t2
    
    prf = []
    hrf = []
    vrf = []
    freq = np.zeros(ns/2)
    
    rfdict= copy.deepcopy(evdict)        
   
    for iev in range(len(rfdict)): 
        rfst = rfdict[iev]['stream']
        #Detrend 
        rfst.detrend(type='demean')
        rfst.detrend(type='linear')
        
        #Filter and pull out components 
        P = bandpass(rfst[2].data, flo, fhi, delta,corners = 4, zerophase = True)
        SV = bandpass(rfst[1].data, flo, fhi, delta,corners = 4, zerophase = True)
        SH = bandpass(rfst[0].data, flo, fhi, delta,corners = 4, zerophase = True)
        
        P *= taper
        SV *= taper
        SH *= taper    
      
        #FFT
        PF = fft.fft(P)
        SVF = fft.fft(SV)
        SHF = fft.fft(SH)    
        frint = delta/ns
        frint = 0.0045
        for k in range(ns/2):
            #fk = k-1
            freq[k]= frint*k
            
        
        #Determine waterlevel 
        PPconj = PF*np.conj(PF)
        for i in range(len(PF)):
            phi[i] = max(PPconj[i], c0*max(PPconj))

        #Gaussian Filter 
        om = freq * 2 * np.pi
        for i in range(len(PF)/2):
            cs[i] = np.conj(1j* om[i]*t0)
            gauf[i] = np.conj(np.exp(-(om[i]/(2*ag))**2))
            if i >=2:
                cs[ns-i+1] = np.conj(cs[i])
                gauf[ns-i+1] = np.conj(gauf[i])
 
    
        PFrf =   (PF * np.conj(PF)) * gauf/phi*np.exp(cs)
        VFrf =   (SVF * np.conj(PF)) * gauf/phi *np.exp(cs)
        HFrf =   (SHF * np.conj(PF)) * gauf/phi *np.exp(cs)

        
        PRF = fft.ifft(PFrf)
        VRF = fft.ifft(VFrf)
        HRF = fft.ifft(HFrf)
        #normalize
        maxrf = max(PRF)
        prf.append(np.real(PRF/maxrf))
        vrf.append(np.real(VRF/maxrf))
        hrf.append(np.real(HRF/maxrf))
    
        #update RF dict, mark which kind of rf we used
        rfdict[iev]['stream'][2].data = prf[iev]
        rfdict[iev]['stream'][1].data = vrf[iev]
        rfdict[iev]['stream'][0].data = hrf[iev]
        rfdict[iev]['rf'] = 'WTR'
    
    return rfdict                    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        