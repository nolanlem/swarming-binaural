import numpy as np
import matplotlib.pyplot as plt
import librosa 
import os 
import glob
import soundfile as sf
os.chdir('/Users/nolanlem/Documents/TEACHING/jupyter-nbs')

#%%
# how can we experiment with timbre using the same sound as a source material? 
# what sound do you think would sound interesting if you heard a lot of them?
# how can we better create a dynamic, sounding environment using just one sound? 
# spatial effects, variations in vol...
# a cricket chirping on it's own isn't that interesting but when you have thousands of them
# interacting, the sound synergizes to create a naturalistic, atmospheric soundscape


#%%
# demonstration of central limit theorem
cl = [] # list to hold sum of random numbers 
numinsum = 10 # how many numbers to generate per sum
numinbatch = 1000 # how many sum of numbers to generate per batch

for i in range(numinbatch):
    a = [np.random.random() for i in range(numinsum)]
    thesum = np.sum(a)/numinsum
    cl.append(thesum)

# #simple density plot 
dist = np.arange(numinbatch)
plt.figure()
ax = plt.gca()
ax.scatter(cl, dist, marker='.')
  
# plot histogram, what shape does this start to resemble? what is the mean centered around? 
plt.figure()  
plt.hist(cl, 50)


#%% now let's apply it to our different sounds to create different volumes for
# each sound source (e.g.cricket)
# max amplitude = 1 without clipping so let's scale down our normal distribution
# and add an offset 
offset = 0.75
scaledown = 4. # 
cl = [elem/scaledown for elem in cl] # 0-0.25
cl = [elem+offset for elem in cl] # later we can worry about 'loudness'

sourcesndfile = "./audio/cricket-chirp-event.wav" # cricket chirp! 
sourcesndfile = '/Users/nolanlem/Documents/kura/kura-python/samples/woodblock_lower.wav'
# what is the original audio file sr? 
y, sr_original = librosa.load(sourcesndfile, sr=22050)


# let's import a sound and then combine multiple sources of them to see what it sounds like 
N = 40 # how many virtual crickets do we want?
b = [[] for elem in range(N)] # create empty array to hold a bunch of instances of the same source sound 


for i in range(N):
    #rand_sr = (0.7 + np.random.uniform(low=0.0, high=0.3))*sr_original
    rand_sr = sr_original
    b[i].append(librosa.load(sourcesndfile, sr=rand_sr)[0]) # just the samples, not the sr
    b[i].append(librosa.load(sourcesndfile, sr=rand_sr)[0]) # just the samples, not the sr

# now our list b holds instances of the sound at different sampling rates 
#%% let's say the crickets are completely surrouding us in a circle 
# let's use hrirs to binaurally place each cricket around us in a sphere
# the left ear HRIRs are in ./LR_hrirs/left, right ear: ./LR_hrirs/right/
# so we'll fill up the buffer twice with each cricket sample
    
L_hrirs = []
R_hrirs = []
allhrirs = []
hrirdir = "./LR_hrirs/"
num_hrirs = 360 # there are 720 hrirs because 360degrees*(l and r ears =2)

for lhrir, rhrir in zip(glob.glob(hrirdir + "left/*.wav"),glob.glob(hrirdir + "right/*.wav")):
    lefthrir, _ = librosa.load(lhrir, sr=sr_original)
    righthrir, _ = librosa.load(rhrir, sr=sr_original)
    allhrirs.append(np.array([lefthrir, righthrir]))
#%% by channel
for i,fi in enumerate(glob.glob(hrirdir + "left/*.wav")):
    L_hrirs.append(librosa.load(fi, sr=cl[i]*sr_original)[0])
    
for i,fi in enumerate(glob.glob(hrirdir + "right/*.wav")):
    R_hrirs.append(librosa.load(fi, sr=cl[i]*sr_original)[0])

deg_inc = float(num_hrirs)/N # its 720
degrees = []
for i in range(N):
    degrees.append(int(i*deg_inc))

#%%
# let's plot where our virtual crickets are located in space
plt.figure()
for d in degrees:
    plt.scatter(np.sin(d), np.cos(d), marker='o')
   
plt.xlabel('horizontal distance')
plt.ylabel('vertical distance')
plt.scatter(0,0, marker='x',label='listener')
plt.legend()
plt.axis('equal')

#%%
# load hrir of acoustic space 
concert_irs = [] # list to hold all IR instances recorded in concert hall in finland 
#concert_ir_sig, _ = librosa.load('./church-ir.wav', mono=False)
concert_ir_sig, _ = librosa.load('./finland-irs/s1_r1_b.wav', mono=False)
for ir in glob.glob('./finland-irs/*.wav'):
    y_, _ = librosa.load(ir, mono=False)
    concert_irs.append(y_)

# now hrtfs holds all the hrtfs that we will convolve with the source audio 
# test 0 and 90 degrees  

outputdir = "conv-outputs-finland/"
if os.path.exists(outputdir) == False:
    os.mkdir(outputdir)
#outputdir = "/Users/nolanlem/Documents/PROJECTS/metronomes/coupled_osc_delay/samples/"

# remove all the outputs in outputdir
for fi in glob.glob(outputdir+"*.wav"):
    os.remove(fi)
# 0 degrees
idx = int(N*(0/360))

def convolve(idx, snd, hrir, snd_ir):
    outputL = np.convolve(snd[0], hrir[0]) # left ear at zero degrees
    outputL = np.convolve(outputL, snd_ir[0]) # convolve with left chan ir of church
    outputR = np.convolve(snd[1], hrir[0]) # right ear at zero degrees
    outputR = np.convolve(outputR, snd_ir[1])           # convolve with right chan of ir church
    return np.array([outputL, outputR])

# def convolve(idx, snd, snd_ir):
#     outputL = np.convolve(s[idx][0], L_hrirs[idx]) # left ear at zero degrees
#     outputL = np.convolve(outputL, yir_l) # convolve with left chan ir of church
#     outputR = np.convolve(s[idx][1], R_hrirs[idx]) # right ear at zero degrees
#     outputR = np.convolve(outputR, yir_r)  
    
conv_out = convolve(0, b[0], allhrirs[0], concert_ir_sig[0])

# now convolve wodlock at degree 0 with concert IR
#sf.write(outputdir+ str(idx) + "_outputstereo.wav", conv_out.T, samplerate=44100)



#%%
# for each cricket file of a slightly different sr, let's convovle each HRIR
# with it
#outputdir = "/Users/nolanlem/Documents/PROJECTS/metronomes/coupled_osc_delay/samples/"


audiobuffer = [] # buffer to hold our convolved hrir cricket audio



for i, deg in enumerate(degrees):
    random_location_int = np.random.randint(low=0, high=len(concert_irs))
    print(i, random_location_int)
    outputStereo = convolve(deg, b[i], allhrirs[deg], concert_irs[random_location_int])
    sf.write(outputdir + str(degrees[i]) + "_output.wav", outputStereo.T, samplerate=22050)
    audiobuffer.append(outputStereo) # save the samples to a buffer



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

#%% let's plot where each of the first 10 crickets are chirping 

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
# plot and listen to first cricket's song
plt.figure()

thecricket = 0
cricket_stereo = np.array([soundbuf_L[thecricket],soundbuf_R[thecricket]]) # to make the stereo file,we have to shape the array into (2,N)
librosa.output.write_wav("./individual-crickets/cricket_" + str(thecricket) + ".wav", cricket_stereo, sr=44100)

# merge the L and R mono tracks to plot our cricket swarm
# typically this is done by summing the columns of the sound buffer (along axis=0)
cricket_merged_mono = np.sum(soundbuf, axis=0)
plt.plot(cricket_merged_mono)


    
#%% write our soundbuf to file! 
librosa.output.write_wav("./cricket-field/" + str(N)+"-cricket-output.wav", soundbuf, sr=44100, norm=True)

#################################################
#%% what if we wanted to synchronize the crickets like they do in real life!? 
#%% let's create a cricket oscillator class
class cricketOscillator():
    def __init__(self, phase, rate, adj, kn):
        self.phase = phase 
        self.rate = rate 
        self.adj = adj
        self.kn = kn
        self.phase_ = []
        self.trig_ = []


def stepForward(crickets):
    for cricket in crickets:
        cricket.phase = cricket.phase + cricket.rate + cricket.adj
        cricket.phase_.append(cricket.phase)


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
        
                

def freq2rad(freq,sr=30):
    return 2*np.pi*freq/sr

def plotPhases(p, R, N):
    plt.figure()
    for ph in p:
        plt.plot(np.sin(ph.phase_), linewidth=0.5)
    R = [(elem)/N for elem in R]
    plt.plot(R)
#%%
# we can treat each cricket like an independent oscilator that is coupled to the others in the group 

N = 2 # 2 crickets
sr = 30 # use a much larger sampling rate  
duration = 5.  # 5 seconds
totaldur = duration*sr 
tsecs = np.linspace(0,duration,totaldur)

sr = 30 # use a much larger sampling rate  
freq1 = 1 
freq2 = 2
# rad vel = 2*pi*f/sr
rate1 = 2*np.pi*freq1/sr
rate2 = 2*np.pi*freq2/sr

iphase1 = 0.0 
iphase2 = np.pi/2

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
        
    
# plot the phases 
plt.figure()
plotPhases(crickets, r, N)
#%%

    
#%% let's create a bunch of crickets with random initial phases and random velocities
N = 100 # 2 crickets
duration = 10.  # 5 seconds
totaldur = duration*sr 
tsecs = np.linspace(0,duration,totaldur)
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

plotPhases(crickets, r, N)

#%%
# let's get the trigger points for each of our crickets from the signal 

audiosr = 44100
triggers = []

# convert triggers to audio sampling rate
for cricket in crickets:
    cricket.trig_ = [int(audiosr*float(elem)/sr) for elem in cricket.trig_]
    triggers.append(cricket.trig_)

#%%  
outputdir = "./cricket-sync-output/"
soundbuf_L = np.zeros((N, int(duration*audiosr) + largestsamps))
soundbuf_R = np.zeros((N, int(duration*audiosr) + largestsamps))

cricketaudio = convolveHRIRS(N, sourcesndfile)

#%%

for i, trigs in enumerate(triggers):
    for trig in trigs:
        soundbuf_L[i][trig:(trig + len(cricketaudio[i][0]))] =  soundbuf_L[i][trig:(trig + len(cricketaudio[i][0]))] + cricketaudio[i][0]
        soundbuf_R[i][trig:(trig + len(cricketaudio[i][1]))] =  soundbuf_R[i][trig:(trig + len(cricketaudio[i][1]))] + cricketaudio[i][1]

L_chan = np.sum(soundbuf_L, axis=0)
R_chan = np.sum(soundbuf_R, axis=0)
stereoOutput = np.array([L_chan, R_chan])

librosa.output.write_wav(outputdir + str(N) + "_crickets-sync.wav", stereoOutput, sr=audiosr, norm=True)

    
 
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
    
    
    
    
    
    
    
    
    
    
    
    
    