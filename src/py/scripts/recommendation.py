
import subprocess
import wave
import struct
import numpy as np
import os
import sys, ast, getopt, types
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.metrics import silhouette_samples, silhouette_score
import json
import io
try:
    to_unicode = unicode
except NameError:
    to_unicode = str

def read_wav(wav_file):
    """ retourne 2 sets wav d'un wavefile"""
    w = wave.open(wav_file)
    n = 60 * 10000
    if w.getnframes() < n * 2:
        raise ValueError('Wave file too short')
    frames = w.readframes(n)
    wav_data1 = struct.unpack('%dh' % n, frames)
    frames = w.readframes(n)
    wav_data2 = struct.unpack('%dh' % n, frames)
    return wav_data1, wav_data2
    
def moments(x):
    mean = x.mean()
    std = x.var()**0.5
    skewness = ((x - mean)**3).mean() / std**3
    kurtosis = ((x - mean)**4).mean() / std**4
    return [mean, std, skewness, kurtosis]

def fftfeatures(wavdata):
    f = np.fft.fft(wavdata)
    f = f[2:(f.size / 2 + 1)]
    f = abs(f)
    total_power = f.sum()
    f = np.array_split(f, 10)
    return [e.sum() / total_power for e in f]

def features(x):
    x = np.array(x)
    f = []

    xs = x
    diff = xs[1:] - xs[:-1]
    f.extend(moments(xs))
    f.extend(moments(diff))

    xs = x.reshape(-1, 10).mean(1)
    diff = xs[1:] - xs[:-1]
    f.extend(moments(xs))
    f.extend(moments(diff))

    xs = x.reshape(-1, 100).mean(1)
    diff = xs[1:] - xs[:-1]
    f.extend(moments(xs))
    f.extend(moments(diff))

    xs = x.reshape(-1, 1000).mean(1)
    diff = xs[1:] - xs[:-1]
    f.extend(moments(xs))
    f.extend(moments(diff))

    f.extend(fftfeatures(x))
    return f


def compute_chunk_features(mp3_file):
    """Return feature vectors for two chunks of an MP3 file."""
    # Extract MP3 file to a mono, 10kHz WAV file
    sox_command = "/usr/local/bin/sox"
    out_file = 'temp.wav'
    #cmd = sox_command % (out_file, mp3_file)
    temp2 = subprocess.call([sox_command, mp3_file,'-r 10000','-c 1',out_file])
    # Read in chunks of data from WAV file
    wav_data1, wav_data2 = read_wav(out_file)
    # We'll cover how the features are computed in the next section!
    return np.array(features(wav_data1)), np.array(features(wav_data2))

#On charge les mp3s

def clustering(files):
  filelist=[]
  featurelist1=[]
  featurelist2=[]
  for file in files:
      try:
          feature_vec1, feature_vec2 = compute_chunk_features(file)
      except:
          print("erreur d'extraction")
          continue
      tail, track = os.path.split(file)
      tail, dir1 = os.path.split(tail)
      title = str(dir1)+'\\\\'+str(track)
      filelist.append(track)
      featurelist1.append(feature_vec1)
      featurelist2.append(feature_vec2)

  FeatNames = ["amp1mean","amp1std","amp1skew","amp1kurt","amp1dmean","amp1dstd","amp1dskew","amp1dkurt","amp10mean","amp10std",
              "amp10skew","amp10kurt",
              "amp10dmean","amp10dstd","amp10dskew","amp10dkurt","amp100mean","amp100std","amp100skew",
              "amp100kurt","amp100dmean","amp100dstd","amp100dskew","amp100dkurt","amp1000mean","amp1000std","amp1000skew",
              "amp1000kurt","amp1000dmean","amp1000dstd","amp1000dskew","amp1000dkurt","power1","power2","power3","power4",
              "power5","power6","power7","power8","power9","power10"]

  #On exporte en csv les data avec les 2 types de features
  datatrain1 = pd.DataFrame(index=filelist,data=np.array(featurelist1),columns=FeatNames)
  datatrain2 = pd.DataFrame(index=filelist,data=np.array(featurelist2),columns=FeatNames)

  data = scale(datatrain1.ix[:,1:])
  reduced_data = PCA(n_components=2).fit_transform(data)
  
  kmeans = KMeans(init='k-means++', n_clusters=13,n_init=500,max_iter=600,n_jobs=2,random_state=10)
  kmeans.fit(reduced_data)

  pred = kmeans.predict(reduced_data)
  groupes = pd.Series(index=datatrain1.index,data=pred, name="cluster")

  Music_clusters = groupes.reset_index().to_dict(orient='records')

  msc = json.dumps(Music_clusters, indent=4)

  print(msc)

def main(argv):            
    arg_dict={}
    switches={'li':list,'di':dict,'tu':tuple}
    singles=''.join([x[0]+':' for x in switches])
    long_form=[x+'=' for x in switches]
    d={x[0]+':':'--'+x for x in switches}
    try:            
        opts, args = getopt.getopt(argv, singles, long_form)
    except getopt.GetoptError:          
        print "bad arg"                       
        sys.exit(2)       

    for opt, arg in opts:        
        if opt[1]+':' in d: o=d[opt[1]+':'][2:]
        elif opt in d.values(): o=opt[2:]
        else: o =''
        if o and arg:
            arg_dict[o]=ast.literal_eval(arg)

        if not o or not isinstance(arg_dict[o], switches[o]):    
            print opt, arg, " Error: bad arg"
            sys.exit(2)                 

    for e in arg_dict:
      clustering(arg_dict[e])

if __name__ == '__main__':
    main(sys.argv[1:])