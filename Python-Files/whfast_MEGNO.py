import rebound
import numpy as np
import time

# Import matplotlib
import matplotlib; matplotlib.use("pdf")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import multiprocessing
import warnings
import csv

def whfast_simulation(par):
    am, em = par # unpack parameters
    
    sim = rebound.Simulation()
    sim.integrator = "whfast"
    sim.ri_whfast.safe_mode = 0
    sim.dt = 0.0005
    sim.units=['Yr2pi', 'AU', 'Msun']
    sim.add(m=1,hash="Sagarmatha")                #Sun-mass star
    sim.add(m=0.0009543,a=1,e=0.36,hash="Laligurans")   #Jupiter-mass planet
    sim.add(m=0.000000036036,a=am,e=em,primary=sim.particles["Laligurans"]) # Satellite around Laligurans
    #m=0.000000036036 Moon-mass
    #m=0.000003003 Earth-mass
    sim.move_to_com()
    
    sim.init_megno()
    sim.exit_max_distance = 20.
    try:
        sim.integrate(100*2.*np.pi, exact_finish_time=0) # integrate to the nearest timestep for each output
        #to keep the timestep constant and preserve WHFast's symplectic nature
        megno = sim.calculate_megno() 
        print("Completed",par)
        return par[0],par[1],megno
    except rebound.Escape:
        print("Completed",par)
        return par[0],par[1],10. # At least one particle got ejected, returning large MEGNO.
		
t1=time.time()
val_megno=whfast_simulation((0.00256,0.1)) 
#val_megno=whfast_simulation((0.00001,0.0001))
t2=time.time()
print("Timer",t2-t1)
print(val_megno)

Ngrid = 40
par_a = np.linspace(0.0,0.04,Ngrid)
par_e = np.linspace(0.,.999,Ngrid)
parameters = []
for e in par_e:
    for a in par_a:
    
        parameters.append((a,e))
from rebound.interruptible_pool import InterruptiblePool
pool = InterruptiblePool()
t1=time.time()
results = pool.map(whfast_simulation,parameters)
t2=time.time()
print("Timer:",(t2-t1))

dt = np.dtype('float,float,float')
data = np.array(results,dtype=dt)
data.dtype.names = ['sm','ecc','megno']
data

### Write to CSV file
filename = 'megno_whfast.csv'    # specify name
with open(filename,'w') as csvfile:
  writer = csv.writer(csvfile,delimiter=',')
  writer.writerow(['Smaxis','InitialEcc','Megno'])
  for index,datum in enumerate(data):
    print(datum[0],datum[1],datum[2])
    writer.writerow([datum[0],datum[1],datum[2]])

    
results2d = np.array(data['megno']).reshape(Ngrid,Ngrid)
#%matplotlib inline
#import matplotlib.pyplot as plt
fig = plt.figure(figsize=(7,5))
ax = plt.subplot(111)
extent = [min(par_a),max(par_a),min(par_e),max(par_e)]
ax.set_xlim(extent[0],extent[1])

ax.set_xlabel("Semi-major axis ($a$)", fontsize=20)
ax.set_ylim(extent[2],extent[3])
ax.set_ylabel("Eccentricity ($e$)", fontsize=20)

plt.yticks([0.2,0.4,0.6,0.8,1.0], fontsize = 20)
plt.xticks([0.0, 0.01, 0.02,0.03,0.04],fontsize = 20)
ax.tick_params(axis='both', direction='in',length = 4.0, width = 4.0)

#ax.axvline(0.023,0.0,1,linestyle='--',linewidth=4,color='white')

im = ax.imshow(results2d, interpolation="none", vmin=1.9, vmax=4, cmap="RdYlGn_r", origin="lower", aspect='auto', extent=extent)
cb = plt.colorbar(im, ax=ax)
cb.set_label("MEGNO $\\langle Y \\rangle$")

#plt.title("Megno map for the system using WHFAST integrator")
fig.savefig("megno_whfast.png",bbox_inches = 'tight')

#fig.savefig("megno-rtbp1.png")


