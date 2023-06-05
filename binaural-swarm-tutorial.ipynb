#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import librosa 
import os 
import glob
import soundfile as sf

os.chdir('/Users/nolanlem/Documents/TEACHING/swarming-binaural/')

#%% make dir if it doesn't exist
def makeDir(dirname):
    if os.path.exists(dirname) == False:
        print('making directory: ', dirname)
        os.mkdir(dirname)

#%%
# how can we experiment with timbre using the same sound as a source material? 
# what sound do you think would sound interesting if you heard a lot of them?
# how can we better create a dynamic, sounding environment using just one sound? 
# spatial effects, variations in vol...
# a cricket chirping on it's own isn't that interesting but when you have thousands of them
# interacting, the sound synergizes to create a naturalistic, atmospheric soundscape


#%% now let's apply it to our different sounds to create different volumes for
# each sound source (e.g.cricket)
# max amplitude = 1 without clipping so let's scale down our normal distribution
# and add an offset 
offset = 0.75
scaledown = 4. # 
cl = [elem/scaledown for elem in cl] # 0-0.25
cl = [elem+offset for elem in cl] # later we can worry about 'loudness'

sourcesndfile = "./audio/cricket-chirp-event.wav" # cricket chirp! 
# sourcesndfile = './audio/woodblock_mono.wav' # or woodblock? 
# what is the original audio file sr? 
y, sr_original = librosa.load(sourcesndfile, sr=22050)


# let's import a sound and then combine multiple sources of them to see what it sounds like 
N = 360 # how many virtual crickets do we want?
b = [[] for elem in range(N)] # create empty array to hold a bunch of instances of the same source sound 


for i in range(N):
    print(f'working on angle {i}/{N}')
    #rand_sr = (0.7 + np.random.uniform(low=0.0, high=0.3))*sr_original
    rand_sr = sr_original
    b[i].append(librosa.load(sourcesndfile, sr=rand_sr)[0]) # just the samples, not the sr
    b[i].append(librosa.load(sourcesndfile, sr=rand_sr)[0]) # just the samples, not the sr

# now our list b holds instances of the sound at different sampling rates 
#%% let's say the crickets are completely surrouding us in a circle 
# let's convolve our hrirs with the source sound to binaurally place each cricket around us in a sphere
# the left ear HRIRs are in ./LR_hrirs/left, right ear: ./LR_hrirs/right/
# so we'll fill up the buffer twice with each cricket sample
    
L_hrirs = []
R_hrirs = []
hrirdir = "./LR_hrirs/"
num_hrirs = 360 # there are 720 hrirs because 360degrees*(l and r ears =2)

Lhrirs = glob.glob(hrirdir + "left/*.wav")
#Lhrirs.sort(key=natural_keys)

Rhrirs = glob.glob(hrirdir + "right/*.wav")
#Rhrirs.sort(key=natural_keys)

for i,fi in enumerate(Lhrirs):
    L_hrirs.append(librosa.load(fi, sr=cl[i]*sr_original)[0])
    
for i,fi in enumerate(Rhrirs):
    R_hrirs.append(librosa.load(fi, sr=cl[i]*sr_original)[0])

deg_inc = float(num_hrirs)/N # its 720
degrees = []
for i in range(N):
    degrees.append(int(i*deg_inc))


#%%
# let's plot where our virtual crickets are located in space
plt.figure(figsize=(5,5))
for d in degrees:
    plt.scatter(np.sin(d), np.cos(d), marker='.')
   
plt.xlabel('horizontal distance')
plt.ylabel('vertical distance')
plt.scatter(0,0, marker='x',label='listener')
plt.legend()
plt.axis('equal')

#%%
# now hrtfs holds all the hrtfs that we will convolve with the source audio 
# test 0 and 90 degrees  
outputdir = "./conv-outputs/"
makeDir(outputdir)
#outputdir = "/Users/nolanlem/Documents/PROJECTS/metronomes/coupled_osc_delay/samples/"

# remove all the outputs in outputdir
for fi in glob.glob(outputdir+"*.wav"):
    os.remove(fi)

# sanity check: let's listen to two of the HRIR positions at 0 and 90 deg (left)
# 0 degrees
idx = int(N*(0/360))
outputL = np.convolve(b[idx][0], L_hrirs[idx]) # left ear at zero degrees
outputR = np.convolve(b[idx][1], R_hrirs[idx]) # right ear at zero degrees
outputStereo = np.array([outputL, outputR]) # create stereo (2, n)
# stereo audio needs to be in columns (len(y),2) so we apply the .T to the array 
sf.write(outputdir+ str(idx) + "_outputstereo.wav", outputStereo.T, samplerate=44100)

# 90 degrees (to LEFT)
idx = int(N*(90/360)) # 
outputL = np.convolve(b[idx][0], L_hrirs[idx]) # left ear at 90 degrees
outputR = np.convolve(b[idx][1], R_hrirs[idx]) # right ear at zero degrees
outputStereo = np.array([outputL, outputR]) # create stereo (2, n)
sf.write(outputdir + str(idx) + "_outputstereo.wav", outputStereo.T, samplerate=44100)


#%% let's combine each HRIR for each ear and hear the whole spatial field (N, 360)

audiosr = 22050
timebetween = np.zeros(int(0.2*audiosr))
bufferL = []
bufferR = []

for lhrir, rhrir in zip(L_hrirs, R_hrirs):
    bufferL.extend(lhrir)
    bufferL.extend(timebetween)
    bufferR.extend(rhrir)
    bufferR.extend(timebetween)


stereoout = np.array([bufferL, bufferR]).T
output = sf.write(outputdir + '/360-spatial-field-hrirs.wav', stereoout, samplerate=audiosr)
# listen with headphones! it's a must, that's how binaural panning works
# what do you notice about the sound 
#%%
# for each cricket file of a slightly different sr, let's convovle each HRIR
# with it
#outputdir = "/Users/nolanlem/Documents/PROJECTS/metronomes/coupled_osc_delay/samples/"
pathforhrirs = os.path.join(outputdir, 'output-by-degree')
if os.path.exists(pathforhrirs) == False:
    os.mkdir(pathforhrirs)

cricket = [] # buffer to hold our convolved hrir cricket audio

for i, deg in enumerate(degrees):

    outputL = np.convolve(b[i][0], L_hrirs[deg]) # left ear at zero degrees
    outputR = np.convolve(b[i][1], R_hrirs[deg]) # right ear at zero degrees    
    outputStereo = np.array([outputL, outputR]) # create stereo (2, n) 
    # write the sound to an output dir
    sf.write(os.path.join(pathforhrirs, str(degrees[i]) + "_output.wav"), outputStereo.T, samplerate=22050)
    cricket.append(outputStereo) # save the samples to a buffer


#%% 
# let's create lists that hold the sample numbers that reflect
# the periodic cricket chirps bounded within a freq range
# we wan
            
duration = 10. # total length of cricket swarm 
sr = 44100 # our sampling rate

# slowest chirp is 0.5 Hz and fastest is 3 Hz
slowrate = sr*1/0.5 
fastrate = sr*1/3.

# generate a uniform distribution within our slow/fast rate limits
randrate = np.random.uniform(low=slowrate, high=fastrate, size=(N,))
randrate = [int(num) for num in randrate] # cast to ints

triggers = [[] for elem in range(N)] # create empty list of lists to hold our sample trigger points

# create a list for each cricket that contains the sample index where they will chirp
# depending on the randomly generated rate above (in randgen)
for i,samp in enumerate(randrate):
    currsamp = samp
    while (currsamp < duration*sr):
        triggers[i].append(int(currsamp))
        currsamp += samp

#%% let's plot where each of the first 10 crickets are chirping as impulses  

num2plot = 10
fig, ax = plt.subplots(num2plot, 1, figsize=(8,8))

for i in range(num2plot):
    ax[i].vlines(triggers[i],-1,1,)
    ax[i].axes.yaxis.set_ticklabels([])
    ax[i].axes.xaxis.set_ticklabels([])

triggers_sec = [format(elem/44100., '.2f') for elem in triggers[-1]]
ax[-1].axes.xaxis.set_ticklabels(triggers_sec)
ax[-1].set_xlabel('sec')



#%% let's assign a different chirping rate for each cricket

# find the longest sample in our cricket buffer so that later we'll know how large 
# to make a 
oldmax = 0
for snd in cricket:
    for chan in snd:
        newmax = len(chan)
        if (newmax > oldmax):
            oldmax = newmax 
largestsamps = oldmax
print ('largest length in our cricket sample buffer is, ', largestsamps)

# create an empty stereo sound buffer of size (2, duration*sr+longestsamps) that we will fill up with our cricket samples triggered at 
# the periodic intervals 
soundbuf = np.zeros((2, int(duration*sr)+largestsamps))

for i, trigs in enumerate(triggers):
    for trig in trigs:
       # print trig
        soundbuf[0][trig:(trig+len(cricket[i][0]))] = soundbuf[0][trig:(trig+len(cricket[i][0]))] + cricket[i][0]
        soundbuf[1][trig:(trig+len(cricket[i][1]))] = soundbuf[1][trig:(trig+len(cricket[i][1]))]+ cricket[i][1]

#%% NB: the other way to do it, create empty buffer of size (N, duration*sr) and then 
# sum columns 
soundbuf_L = np.zeros((N, int(duration*sr) + largestsamps))
soundbuf_R = np.zeros((N, int(duration*sr) + largestsamps))

for i, chirp in enumerate(triggers):
    for trig in trigs:
        soundbuf_L[i][trig:(trig + len(cricket[i][0]))] =  soundbuf_L[i][trig:(trig + len(cricket[i][0]))] + cricket[i][0]
        soundbuf_R[i][trig:(trig + len(cricket[i][1]))] =  soundbuf_R[i][trig:(trig + len(cricket[i][1]))] + cricket[i][1]



#%%
makeDir("./individual-crickets/") # make a directory for the individual crickets 
# plot and listen to first cricket's song
plt.figure(figsize=(9,5))

thecricket = 0
cricket_stereo = np.array([soundbuf_L[thecricket],soundbuf_R[thecricket]]) # to make the stereo file,we have to shape the array into (2,N)
sf.write("./individual-crickets/cricket_" + str(thecricket) + ".wav", cricket_stereo.T, samplerate=44100)

# merge the L and R mono tracks to plot our cricket swarm as a mono wf
# typically this is done by summing the columns of the sound buffer (along axis=0)
cricket_merged_mono = np.sum(soundbuf, axis=0)
plt.plot(cricket_merged_mono)
xticks = np.arange(0,len(cricket_merged_mono),44100)
plt.gca().set_xticks(xticks)
plt.gca().set_xticklabels([elem/44100 for elem in xticks]) # show seconds instead of samples for x axis
plt.gca().set_xlabel('seconds')

# write our soundbuf to file! 
makeDir('./cricket-field/')
sf.write("./cricket-field/" + str(N)+"-cricket-output.wav", soundbuf.T, 44100)

#################################################

#%% ###### Using a Coupled Oscillator Model to Synchronize Chirps #####
###### what if we wanted to synchronize the crickets like they do in real life! 
#########################################
#%% let's create a cricket oscillator class
class cricketOscillator():
    def __init__(self, phase, rate, adj, kn):
        self.phase = phase 
        self.rate = rate 
        self.adj = adj
        self.kn = kn
        self.phase_ = []
        self.trig_ = []

# iterate forward 
def stepForward(crickets):
    for cricket in crickets:
        cricket.phase = cricket.phase + cricket.rate + cricket.adj
        cricket.phase_.append(cricket.phase)


# function to get complex order parameters (phase coherence)
def getOrderParams(crickets):
    x = 0 # real part 
    y = 0 # imag part
    for cricket in crickets:
        x += np.cos(cricket.phase)
        y += np.sin(cricket.phase)
        
    meanfield_angle = np.arctan2(y,x)
    meanfield_mag = np.sqrt(x*x + y*y)
   
    return meanfield_mag, meanfield_angle

def couple(crickets, R, psi):
    for cricket in crickets:
        cricket.adj = R*cricket.kn*np.sin(psi-cricket.phase)/N

def checkZeroCrossing(iteration, crickets):
    for cricket in crickets:
        if(cricket.phase >= 2*np.pi):
            cricket.phase = cricket.phase % (2*np.pi)
            cricket.trig_.append(iteration)
        
                
# function to convert frequency to radians
def freq2rad(freq,sr=30):
    return 2*np.pi*freq/sr

def plotPhases(p, R, N, ax):
    for ph in p:
        plt.plot(np.sin(ph.phase_), linewidth=0.5)
    R = [(elem)/N for elem in R]
    ax.plot(R, label='Phase Coherence')
#%%
# we can treat each cricket like an independent oscilator that is coupled to the others in the group 

N = 2 # 2 crickets
sr = 30 # dt --> larger sampling rate produces a better approximation of the integration  
duration = 5.  # 5 seconds
totaldur = duration*sr # total duration to let play out
tsecs = np.linspace(0,duration,int(totaldur))

sr = 30 # use a much larger sampling rate  
freq1 = 1 # frequency of cricket 1
freq2 = 2 # '' cricket 2
rate1 = 2*np.pi*freq1/sr # freq -> rate
rate2 = 2*np.pi*freq2/sr

iphase1 = 0.0 # init. phase of cricket 1
iphase2 = np.pi/2 # init. phase of cricket 2

kn = 0.12 # coupling, same for both crickets

# let's create two instances of our cricketOscillator Class (phase, rate, adj, kn)
cricket1 = cricketOscillator(iphase1, rate1, 0, kn)
cricket2 = cricketOscillator(iphase2, rate2, 0, kn)

crickets = [cricket1, cricket2]
r = []
psi = []

p = [[] for elem in range(N)]

for i in range(int(totaldur)):
    stepForward(crickets)
    R, avgang = getOrderParams(crickets)
    couple(crickets, R, avgang) 
    checkZeroCrossing(i, crickets)
    # save the phases/phasor
    r.append(R)
    psi.append(psi)
        
    
# plot the phases and phase coherence 
plt.figure()
ax = plt.gca()
plt.xlabel('iteration')
plt.ylabel('phase [-1,1]')
plotPhases(crickets, r, N, ax)
plt.legend()

    
#%% let's create a bunch of crickets with random initial phases and random velocities
N = 100 # 2 crickets
duration = 10.  # 5 seconds
totaldur = duration*sr 
tsecs = np.linspace(0,duration,int(totaldur))
kn = 0.35

crickets = []
r = []
psi = []

for i in range(N):
    iphase = np.random.uniform(low=0, high=2*np.pi)
    freq = np.random.uniform(low=0.5, high=3.0)
    rate = freq2rad(freq)
    crickets.append(cricketOscillator(iphase, rate, 0, kn))

        
p = [[] for elem in range(N)]

for i in range(int(totaldur)):
    stepForward(crickets)
    R, avgang = getOrderParams(crickets)
    couple(crickets, R, avgang)    
    checkZeroCrossing(i, crickets)

    # save the phases/phasor
    r.append(R)
    psi.append(avgang)

plt.figure(figsize=(10,7))
ax = plt.gca() # get the axis from plt.figure
plotPhases(crickets, r, N, ax)
plt.legend()


#%%
# let's get the trigger points for each of our crickets from the signal 

audiosr = 44100
triggers = []

# convert triggers to audio sampling rate
for cricket in crickets:
    cricket.trig_ = [int(audiosr*float(elem)/sr) for elem in cricket.trig_]
    triggers.append(cricket.trig_)

#%% given N oscillators, this function distributes them as sound sources around a circle 
#   by using the appropriate HRIR and convolving an audiofile with it to produce spatialization 
def convolveHRIRS(N, sndfile="./audio/cricket-chirp-event.wav"):
    # what is the original audio file sr? 
    y, sr_original = librosa.load(sndfile)
    
    # let's import a sound and then combine multiple sources of them to see what it sounds like 
    b = [[] for elem in range(N)] # create empty array to hold a bunch of instances of the same source sound 

    # load a buffer of size (2, len(sndfile))
    for i in range(N):
        randomrate = 0.7 + np.random.uniform(low=0.0, high=0.3)       
        b[i].append(librosa.load(sndfile, sr=(randomrate)*sr_original)[0]) # just the samples, not the sr
        b[i].append(librosa.load(sndfile, sr=(randomrate)*sr_original)[0]) # just the samples, not the sr
        
    cricketaudio = [] # buffer to hold our convolved hrir cricket audio
    
    deg_inc = float(720)/N # there are 720 hrir files
    degrees = []
    
    for i in range(N):
        degrees.append(int(i*deg_inc))
    
    for i, deg in enumerate(degrees):
        outputL = np.convolve(b[i][0], L_hrirs[deg]) # left ear at zero degrees
        outputR = np.convolve(b[i][1], R_hrirs[deg]) # right ear at zero degrees    
        outputStereo = np.array([outputL, outputR]) # create stereo (2, n) 
        # write the sound to an output dir
        #librosa.output.write_wav(outputdir + str(degrees[i]) + "_output.wav", outputStereo, sr=44100, norm=True)
        cricketaudio.append(outputStereo)
    return cricketaudio   
    

#%%  
outputdir = "./cricket-sync-output/" # output directory to hold the self-synchronizing cricket swarm
makeDir(outputdir)

soundbuf_L = np.zeros((N, int(duration*audiosr) + largestsamps)) # L chan empty audio buffer
soundbuf_R = np.zeros((N, int(duration*audiosr) + largestsamps)) # R chan empty audio buffer

#cricketaudio = convolveHRIRS(N, sndfile=sourcesndfile)
sndfile = sourcesndfile

y, sr_original = librosa.load(sndfile) # load source sound

# let's import a sound and then combine multiple sources of them to see what it sounds like 
b = [] # create empty array to hold a bunch of instances of the same source sound for L and R channels

# load up buffer with sound file at a  slightly random samprate to slightly change the pitch of each chirp 
for i in range(N):
    randomrate = 0.7 + np.random.uniform(low=0.0, high=0.3)       
    b.append(librosa.load(sndfile, sr=(randomrate)*sr_original)[0]) # just the samples, not the sr
        
cricketaudio = [] # buffer to hold our convolved hrir cricket audio

deg_inc = float(360)/N # there are 360 hrir files per chan
degrees = []

# load up degree positions
for i in range(N):
    degrees.append(int(i*deg_inc))

for i, deg in enumerate(degrees):
    outputL = np.convolve(b[i], L_hrirs[deg]) # left ear at zero degrees
    outputR = np.convolve(b[i], R_hrirs[deg]) # right ear at zero degrees    
    outputStereo = np.array([outputL, outputR]) # create stereo (2, n) 
    # write the sound to an output dir
    #librosa.output.write_wav(outputdir + str(degrees[i]) + "_output.wav", outputStereo, sr=44100, norm=True)
    cricketaudio.append(outputStereo)

#%fill up the empty buffers with the binaural chirps at the trigger points

for i, trigs in enumerate(triggers):
    for trig in trigs:
        soundbuf_L[i][trig:(trig + len(cricketaudio[i][0]))] =  soundbuf_L[i][trig:(trig + len(cricketaudio[i][0]))] + cricketaudio[i][0]
        soundbuf_R[i][trig:(trig + len(cricketaudio[i][1]))] =  soundbuf_R[i][trig:(trig + len(cricketaudio[i][1]))] + cricketaudio[i][1]

L_chan = np.sum(soundbuf_L, axis=0)
R_chan = np.sum(soundbuf_R, axis=0)
stereoOutput = np.array([L_chan, R_chan])

## write the audio file to the outputdir
sf.write(os.path.join(outputdir, str(N) + "_crickets-sync.wav"), stereoOutput.T, samplerate=44100)

## end 
 
#%%%%   

# demonstration of central limit theorem
cl = [] # list to hold sum of random numbers 
numinsum = 10 # how many numbers to generate per sum
numinbatch = 1000 # how many sum of numbers to generate per batch

for i in range(numinbatch):
    a = [np.random.random() for i in range(numinsum)]
    thesum = np.sum(a)/numinsum
    cl.append(thesum)

# simple density plot 
dist = np.arange(numinbatch)
plt.figure()
ax = plt.gca()
ax.scatter(cl, dist, marker='.')
  
# plot histogram, what shape does this start to resemble? what is the mean centered around? 
plt.figure()  
plt.hist(cl, 50)
plt.ylabel('count')
plt.xlabel('bin')
 

    
    
    
    
    
    
    