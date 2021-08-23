# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 12:47:03 2018

@author: Jackie Smale and Jan Dettmer

This script will download data for a time window for a specified station from IRIS. Requires 
"""

import sys
sys.path.append("/home/jackie/src_rf")
from numpy import linspace
import direct_stack
import rf_functions
import cPickle as pkl
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import correlate
from os import path, chdir, mkdir
from mpl_toolkits.mplot3d import Axes3D
from rf import rfstats, RFStream
from obspy.core import UTCDateTime, read, Stream
from obspy.core.event import read_events
from obspy.core.inventory import read_inventory, Channel
from obspy.clients.fdsn.client import Client
from mpl_toolkits.basemap import Basemap, cm 
from windrose import WindroseAxes
import copy 

#############################################################
################################################################################
##
##  Start of parameters
##
statn  = 'WHY'
netwrk = 'CN'
workdir = '/home/jackie/rf_studies_2/'  #Path to your RF directory 


# define reference phase to place window around; usually 'P' or 'S'
phase = 'P' 
maxevents = 10000;  # Max number of events

## Pull catalog & data between these dates
t1data = UTCDateTime("2016-08-24T10:00:00")
t2data = UTCDateTime("2016-08-24T11:00:00")

# Time window around phase to request data for
interval = [-100.,300.] #can cut this down! 400 is a lot, be careful not to use first 200s

# Bandpass frequencies
#fmin = 0.05 
#fmax = 1.5
fmin = 0.005
fmax = 15.
newrange = [30,180] #time window to cut out from original (raw) trace; in sec

# set to true to integrate traces, i.e. to work on displacement instead of velocity traces
displacement = False 

## These crieteria allow us to choose a subset of events from downloaded catalog (catlg2):
magrange  = [6.5,7.5]
bazrange  = [0.,360.]
slowrange = [0.01,0.10]
distrange = [30.,90.]
deprange  = [0.,700.]
rotate = 'ZRT' #ZRT, LQT or PVH coordinate system to rotate into
islow = 'True'
rfpros = 'WTR' #WTR for waterlevel WNR for Audet's2010
snrzcut = 5
snrnecut = 1
alpha = 6
beta = 3.5
##
rflen = 50
parr = 70

##
##  End of parameters
##

################################################################################
################################################################################
# Names the data folder after the station. All output saved here.
outfolder = workdir + statn +  '/'

try:
  mkdir(outfolder)
except: #exists already
  pass

################################################################################
################################################################################
## Get station inventory from online repository.
##
client = Client("IRIS")
#starttime = UTCDateTime("1989-01-01")
starttime = t1data
endtime = t2data
#endtime = UTCDateTime("2018-01-02")
finventoryxml = outfolder + statn + '.xml'
if not path.isfile(finventoryxml):
  inventory = client.get_stations(network=netwrk, station=statn,channel="BH*,LH*",
                                  starttime=starttime,
                                  endtime=endtime,
                                  level='channel')
#  stn=inventory[0][0]
#  cha = Channel(code="BHZ",location_code="--",
#    latitude=stn.latitude,longitude=stn.longitude,elevation=stn.elevation,
#    depth=0.0,azimuth=0.0,dip=-90.0,sample_rate=40)
#  stn.channels.append(cha)
#  net=inventory[0]
#  net.stations = [stn]
#  inventory.networks = [net]
  
  inventory.write(finventoryxml, format="stationxml", validate=True)
else:
  inventory = read_inventory(finventoryxml, format="STATIONXML")

net=inventory[0]
stn=inventory[0][0]
cha=inventory[0][0][0]
seed_id = net.code+'.'+stn.code+'.'+cha.location_code+'.'+cha.code
stn_coords = inventory.get_coordinates(seed_id)
statlat = stn.latitude
statlon = stn.longitude


#move to the top, include Dnotedit statement
pvh = 0
if rotate == 'PVH':
    rotate = 'ZRT'
    pvh = 1
################################################################################
################################################################################
##
## step 1: Request catalog from iris, request data for catalog, save to directory
##
fcatalogxml = outfolder + statn + '_catalog.xml'
if not path.isfile(fcatalogxml):
  catlg = client.get_events(latitude=stn.latitude,longitude=stn.longitude,
                            minradius=distrange[0],maxradius=distrange[1],mindepth=deprange[0],
                            maxdepth=deprange[1], starttime = t1data, endtime = t2data,
                            minmagnitude = magrange[0], maxmagnitude=magrange[1])
  ## Save updated catalog with only those events where waveforms are available:
  catlg2 = direct_stack.cut_events(catlg,stn,net,outfolder,workdir,
          interval=interval,phase=phase,Nmax=maxevents) #changing phase to S should work
  catlg2.write(fcatalogxml, format="QUAKEML")
else:
  ## Load catalog with only those events where waveforms were available:
  catlg2 = read_events(fcatalogxml, format="QUAKEML")

################################################################################
################################################################################
##
## step 2:pre-processing, event selection
##
feventpickle = outfolder + statn + 'eventdict.pickle'

if path.isfile(feventpickle):
      print 'Pickle file for station exists. Loading pickle file.'
      with open(feventpickle, 'rb') as handle:
        eventdict = pkl.load(handle)
else:
      print 'Event pickle file does not exist. Start processing eventdict for ',maxevents,' events.'
      eventdict, eventdict_unrot = direct_stack.prepare_traces(outfolder,catlg2,fmin=fmin,fmax=fmax,cutaway=newrange,
                  rotate=rotate,displ=displacement,phase=phase, Nmax=maxevents)
      snr_z = np.zeros(len(eventdict))
      snr_ne = np.zeros(len(eventdict))
      ev_iev = np.ones(len(eventdict))
      [snr_ne, snr_z] = rf_functions.snr_calc(eventdict_unrot, rflen= rflen, parr=parr)
      olen = len(eventdict)
      #get rid of entries with low snr 
      for iev in range(olen) :
        key = 'stream'
        if snr_z[iev] < snrzcut or snr_ne[iev] <snrnecut:
            ev_iev[iev] = 0
            
        #CPY DICT            
      eventdict_chq = copy.deepcopy(eventdict)   
      
      
      print 'Selecting events according to criteria...'
      #PLOTTING SECTION!
      #plt.figure()
      #plt.plot(eventdict[0]['stream'][2].data)
      #plt.title('Eventdict')
  
      #plt.figure()
      #plt.plot(eventdict_chq[0]['stream'][2].data)
      #plt.title('Eventdict chq')  
  
      #plt.figure()
      #plt.plot(eventdict_unrot[0]['stream'][2].data)
      #plt.title('Eventdict_unrot') 
      evdict_new = rf_functions.event_select(eventdict,ev_iev,fmin,fmax,magrange=magrange,bazrange=bazrange,
             distrange=distrange,deprange=deprange,slowrange=slowrange,islow=islow)
      for iev in range(len(evdict_new)):
          evdict_new[iev]['olen'] = olen #length of original evdict before selection
          
          
          
      print 'Saving eventdict to pickle file ',feventpickle,'.'    
      if pvh == 1:
         eventdict =rf_functions.rot_rtz2psv(evdict_new, beta =beta, alph = alpha)   
      with open(feventpickle, 'wb') as handle:
        pkl.dump(eventdict, handle, protocol=pkl.HIGHEST_PROTOCOL)


#plt.figure()
#plt.plot(eventdict[0]['stream'][2].data)
#plt.title('Eventdict tgubg')
  
#plt.figure()
#plt.plot(eventdict_unrot[0]['stream'][2].data)
#plt.title('Eventdict_unrot thgu')
     
################################################################################
     


print 'RF processing...'
frfpickle = outfolder + statn + 'rfdict.pickle'

if rfpros == 'WTR':
    rfdict = rf_functions.process_rf(eventdict,stn_coords,Nmax=maxevents,stn=stn)
    print 'Saving eventdict to pickle file ',frfpickle,'.'
    with open(frfpickle, 'wb') as handle:
        pkl.dump(rfdict, handle, protocol=pkl.HIGHEST_PROTOCOL)
if rfpros == 'WNR':
    rfdict = rf_functions.wiener_rf(eventdict, rflen= rflen, parr=parr, tapelen=50, flo=0.1, fhi=1.0)
    print 'Saving eventdict to pickle file ',frfpickle,'.'
    with open(frfpickle, 'wb') as handle:
        pkl.dump(rfdict, handle, protocol=pkl.HIGHEST_PROTOCOL)
        
###############################################################################


#PLOTTING SECTION!
plt.figure()
plt.plot(eventdict[0]['stream'][2].data)
plt.title('Eventdict 2')
  
plt.figure()
plt.plot(eventdict_chq[0]['stream'][2].data)
plt.title('Eventdict chq 2')  
  
plt.figure()
plt.plot(eventdict_unrot[0]['stream'][2].data)
plt.title('Eventdict_unrot2') 

#PLOT ALL THE BAZ, SLOW, DIST info        
print 'Plotting selected event baz and slowness...'
#set up plot for baz vs. dist/slowness
NEV = len(eventdict)
slow = np.zeros(NEV)
baz = np.zeros(NEV)
dist = np.zeros(NEV)
depth = np.zeros(NEV)

for iev in eventdict:
    event = eventdict[iev]
    slow[iev] = event.get('slowness')
    baz[iev] =  event.get('baz')
    dist[iev] =  event.get('distance')
    depth[iev] =  event.get('depth')

plt.figure()
cm = plt.cm.get_cmap('seismic')
plt.subplot(311)
plt.scatter(x=slow,y=baz,c=dist,cmap=cm)
cb1=plt.colorbar()
cb1.set_label('Distance (deg.)')
plt.xlabel('Slowness (s/km)')
plt.ylabel('BAZ (deg.)')

plt.subplot(312)
plt.scatter(x=dist,y=baz,c=slow,cmap=cm)
cb2=plt.colorbar()
cb2.set_label('Slowness (s/km)')
plt.xlabel('Distance (deg.)')
plt.ylabel('BAZ (deg.)')

plt.subplot(313)
plt.scatter(x=dist,c=slow,y=depth,cmap=cm)
cb3=plt.colorbar()
cb3.set_label('Slowness (s/km)')
plt.xlabel('Distance (deg.)')
plt.ylabel('Depth (km)')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(dist, slow, depth)
ax.set_xlabel('Distance (deg.)')
ax.set_ylabel('Slowness (s/km)')
ax.set_zlabel('Depth (km)')
ax.invert_zaxis()

fig = plt.figure()
plt.hist2d(baz,slow)
plt.xlabel('BAZ (deg)')
plt.ylabel('Slow (s/km)')
cb4 = plt.colorbar()


mapfig = plt.figure()
lon = np.zeros(NEV)
lat = np.zeros(NEV)
for iev in rfdict:
    event =rfdict[iev]
    lon[iev] = eventdict[iev]['event']['origins'][0]['longitude']
    lat[iev] = eventdict[iev]['event']['origins'][0]['latitude']
m = Basemap(projection='aeqd',lat_0=statlat,lon_0= statlon,resolution='l')
m.drawcoastlines(linewidth=0.25)
m.fillcontinents(color = 'coral')
x, y = m( lon , lat)
m.scatter(x,y, 3, marker = '^')
xpt, ypt = m(statlon, statlat)
m.plot([xpt],[ypt],'k^')
m.drawparallels(np.arange(-80,81,20), xoffset=xpt, yoffset=ypt)
m.drawmeridians(np.arange(-180,180,20), xoffset=xpt, yoffset=ypt)


ax = WindroseAxes.from_ax()
ax.bar(baz, dist, opening=0.8, edgecolor='white') 
ax.set_legend()



# PLOT THE TRACES
trsv = []
trsh = []
trp = []

for event in eventdict:
  tr0=eventdict[0].get("stream")
  tr=eventdict[event].get("stream")
  trsh.append(tr[0])
  trsv.append(tr[1])
  trp.append(tr[2])

radall = plt.figure()
plt.title('SV Traces all events')
time_stk = trsv[0].times()
trsv_stk = np.sum([trev.data for trev in trsv], axis=0)/len(trsv)
for tr in trsv:
  plt.plot(time_stk,tr.data,color=[.8,.8,.8])
plt.plot(time_stk,trsv_stk,linewidth=2.0)

tranall = plt.figure()
plt.title('SH Traces all events')
trsh_stk = np.sum([trev.data for trev in trsh], axis=0)/len(trsh)
for tr in trsh:
  plt.plot(time_stk,tr.data,color=[.8,.8,.8])
plt.plot(time_stk,trsh_stk,linewidth=2.0)

vertall = plt.figure()
plt.title('P traces all events')
trp_stk = np.sum([trev.data for trev in trp], axis=0)/len(trp)
for tr in trp:
  plt.plot(time_stk,tr.data,color=[.8,.8,.8])
plt.plot(time_stk,trp_stk,linewidth=2.0)


# PLOT THE RFs
rfsv = []
rfsh = []
rfp = []

for event in rfdict:
  rf0=rfdict[0].get("stream")
  rf=rfdict[event].get("stream")
  xc=correlate(rf0[2].data[0:-1],rf[2].data[0:-1], mode='same')
  lag = np.argmax(xc)- len(xc)/2
  rf[0].data = np.roll(rf[0].data, shift=int(np.ceil(lag)))
  rf[1].data = np.roll(rf[1].data, shift=int(np.ceil(lag)))
  rf[2].data = np.roll(rf[2].data, shift=int(np.ceil(lag)))
  t1=rf[0].stats.starttime
  rfsh.append(rf[0])
  rfsv.append(rf[1])
  rfp.append(rf[2])

radall = plt.figure()
plt.title('SV RF all events')
time_stk = rfsv[0].times()
rfsv_stk = np.sum([tr.data for tr in rfsv], axis=0)/len(rfsv)
for tr in rfsv:
  plt.plot(time_stk,tr.data,color=[.8,.8,.8])
plt.plot(time_stk,rfsv_stk,linewidth=2.0)

tranall = plt.figure()
plt.title('SH RF all events')
rfsh_stk = np.sum([tr.data for tr in rfsh], axis=0)/len(rfsh)
for tr in rfsh:
  plt.plot(time_stk,tr.data,color=[.8,.8,.8])
plt.plot(time_stk,rfsh_stk,linewidth=2.0)

vertall = plt.figure()
plt.title('P RF all events')
rfp_stk = np.sum([tr.data for tr in rfp], axis=0)/len(rfp)
for tr in rfp:
  plt.plot(time_stk,tr.data,color=[.8,.8,.8])
plt.plot(time_stk,rfp_stk,linewidth=2.0)
