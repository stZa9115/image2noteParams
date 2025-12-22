from notation_to_parameters import *
import librosa
import soundfile as sf
import math
import json
from scipy.signal import *
import os

plot = False

full_note_len = 2.4
quick_atk = True
atk_peak = 1.4
release_len = 4000

vibration_alpha = True
vibration_mag = False
vib_phase = [5.53680987, 2.84918514, 2.82814862, 0.67653421, 5.38450552, 1.89833481, 5.0496042, 6.27392924, 2.78018538, 3.57957282, 0.54021761, 2.11627307, 0.09929233510743152, 1.5844700593838439, 1.7401402457810882, 1.6829209340242632, 1.4616168312000724, 2.2795164051358645, 2.624515196610838, 1.1805690352180205, 2.6623809176263844, 2.430137828731036, 0.30124284719034305, 3.7108406595235, 0.0016829209340242632, 3.084794072066474, 1.4296413334536116, 5.580565817224456, 5.291103416572283, 0.9575820114598058, 1.018167165084679, 4.384009033133205, 0.4745837033948421, 1.7283597992429183, 0.6151076013858682, 1.036679295358946, 1.2950076587316706, 1.7796888877306583, 1.3076295657368524, 0.7573144203109184, 5.010055620590231, 0.4956202150701455, 4.449642949560152, 5.509883137995438, 2.9005142297908173, 1.0526670442321766, 1.6492625153437777, 1.931993232259854, 5.138799072043088, 0.8010703645955491]
vib_phase = [0,          3.59556058, 3.57452406, 1.42290965, 6.13088096, 2.64471025, 5.7959797, 0.73711937, 3.52656082, 4.32594826, 1.28659305, 2.86264851, 0.84566777, 2.33084549, 2.48651568, 2.42929637, 2.20799227, 3.02589184, 3.37089063, 1.92694447, 3.40875635, 3.17651326, 1.04761828, 4.45721609, 0.74805836, 3.83116951, 2.17601677, 0.04375594, 6.03747885, 1.70395745, 1.7645426,  5.13038447, 1.22095914, 2.47473523, 1.36148304, 1.78305473, 2.04138309, 2.52606432, 2.054005,   1.50368985, 5.75643105, 1.24199565, 5.19601838, 6.25625857, 3.64688966, 1.79904248, 2.39563795, 2.67836867, 5.88517451, 1.5474458 ]
vib_amp = [0.14357095774419332, 0.33635365095684117, 0.25391482481587757, 0.05075399254764121, 0.3930651351648941, 0.36554165678333406, 0.22940795822238047, 0.2530248819095757, 0.6606079930461508, 0.2602258452265681, 0.08604641553385403, 0.5770706306058275, 0.2629444325911669, 0.5156180427162119, 0.15290505902344448, 0.5549476331157179, 0.2653223650354921, 0.3666558256016987, 0.5467630142249739, 0.2482521946260533, 0.2304233258784584, 0.428719925169106, 0.23261003048772377, 0.15640063695322717, 0.11359505673365823, 0.2275863265737768, 0.13665261055951763, 0.12528825595253473, 0.20302297261717295, 0.1601859462826407, 0.28574663084587243, 0.10306470546579707, 0.16114502041667744, 0.10052206098827093, 0.05243605528626826, 0.0913077872686589, 0.1311037322678462, 0.07746522436760915, 0.23096392136116584, 0.3742851037714845, 0.33640400961888883, 0.21873872920014595, 0.1886627342798931, 0.28548572060658534, 0.061401591160784155, 0.09557714451423308, 0.3027531086955034, 0.32617508941824996, 0.16107082832346206, 0.14403599690235006]
# vib_amp = [0.004387044282123212, 0.008028885758092675, 0.0024723642804005963, 0.003351917008762816, 0.0065980919368036055, 0.0040756521015001546, 0.0019150360914909912, 0.0006805338980789058, 0.0013240639511691537, 0.0002892146027922755, 6.574529134912167e-05, 0.0005246572510019978, 0.0003824681844240474, 0.0009143299172050327, 0.00015652795551135775, 0.00024220135307101388, 0.0002204727952057935, 8.180619937182346e-05, 9.987636027550228e-05, 5.579105877045497e-05, 5.837040945413734e-05, 5.8536217467200426e-05, 1.8365839854847652e-05, 3.1868117240515284e-05, 7.5885793334525426e-06, 1.808304389092854e-05, 2.2985958502444482e-05, 1.7165867142425264e-05, 1.6146095162022526e-05, 1.5403935535413787e-05, 2.0403820773289243e-05, 1.4972721759351218e-05, 1.2479397918173875e-05, 8.488453338350872e-06, 7.176441318455684e-06, 5.908493312332547e-06, 4.9579962648387e-06, 4.323055715499605e-06, 4.972635163415998e-06, 4.2133995474395384e-06, 4.809327867667732e-06, 3.5996672030791224e-06, 1.8864605425815022e-06, 1.6460476605084939e-06, 1.8132999945290033e-06, 1.9876378825124272e-06, 2.608153548075658e-06, 1.88551230011061e-06, 8.951831101108585e-07, 8.08557365289395e-07]
d_vib = 5
d_range = 5
d_freq = 3
thres = 0.5

timbre_gain = 1.7
shelving_filter_gain = 1.4


def timesfilt(b, a, signal, times):
    output = signal
    for _ in range(round(times / 2)):
        output = filtfilt(b, a, output)
    return output
    
def poly_fit(y, d=20):
    t = np.arange(y.shape[0]) / y.shape[0]
    z = np.polyfit(t, y, deg=d)
    p = np.poly1d(z)
    res = y - p(t)
    plt.plot(y)
    plt.plot(p(t))
    if plot:
        plt.show()
        plt.plot(res)
        plt.show()
    return p(t), res
    
    
def linearReSample(y, target_sr):
    l = y.shape[0] - 1
    out = np.zeros(shape=(target_sr,))
    for i in range(target_sr):
        pos = i / target_sr * l
        floor = int(math.floor(pos))
        cur_diff = (y[floor+1] - y[floor])
        out[i] = y[floor] + cur_diff * (pos - floor)
    return out

def linearReSample2(y, target_sr):
    l = y.shape[0]
    tmp = librosa.resample(y, orig_sr=l, target_sr=(l * 100), res_type='linear')
    out = librosa.resample(tmp, orig_sr=tmp.shape[0], target_sr=target_sr, res_type='linear')
    if out.shape[0] != target_sr:
        out = out[:target_sr]
    return out

def concat(atk, sus, ovlp, mod='hann'):
    split = [0] * 4
    split[1] = atk.shape[0]
    split[0] = split[1] - ovlp
    
    length = atk.shape[0] + sus.shape[0] - ovlp
    out = np.zeros(shape=(length,))
    lin_env = np.arange(ovlp) / ovlp
    out[:split[0]] = atk[:split[0]]
    if mod == 'linear':
        out[split[0]:split[1]] = atk[split[0]:] * (1 - lin_env) + sus[:ovlp] * lin_env
    elif mod == 'hann':
        end_env = 0.5 * (1 + np.cos(np.pi*lin_env))
        start_env = 0.5 * (1 + np.cos(np.pi*(1 - lin_env)))
        out[split[0]:split[1]] = atk[split[0]:] * end_env + sus[:ovlp] * start_env
    out[split[1]:] = sus[ovlp:]
    return out

def vib_pars(vib, fs):
    state = 'rising'
    localpeak = []
    localbot = []
    l = vib.shape[0]
    for i in range(l-1):
        if vib[i+1] < vib[i] and state == 'rising':
            state = 'falling'
            localpeak.append(i)
        elif vib[i+1] > vib[i] and state == 'falling':
            state = 'rising'
            localbot.append(i)
            
    vib_range = np.zeros(shape=vib.shape)
    vib_freq = np.zeros(shape=vib.shape)
    vib_freq_mean = []
    ovlp = 4000
    
    for i in range(len(localpeak)-1):
        stamp = localpeak[i]
        stamp2 = localpeak[i+1]
        stamp_mid = round((stamp + stamp2)/2)
        amp = (vib[stamp] - vib[localbot[i]]) / 2
        if amp < thres:
            if localpeak[i] < ovlp:
                continue
            vib_range[stamp:] = 0
            vib_freq[stamp:] = vib_freq[stamp-1]
        else:
            vib_range[stamp:] = amp
            vib_freq[stamp:] = fs / (stamp2 - stamp)
            vib_freq_mean.append(fs / (stamp2 - stamp))
    
    return vib_range, vib_freq, np.mean(vib_freq_mean)

def release_decay(freq):
    const_exp = 0.998
    var_exp = 1 - const_exp
    return const_exp + var_exp * np.power(1 - freq / 22050, 3)


def note_construct(expr, des):
    note = 65
    seq = 1
    note_trend = np.array(expr[1])
    r = (np.max(note_trend) - np.min(note_trend))
    if  r > 1:
        note_trend = (note_trend - round(np.average(note_trend))) / r
    note_trend += note
    pitch = 440 * np.power(2, (note_trend-69)/12)
    
    with open(f'./table/long/note{note}-{seq}.json') as jf:
        pars = json.load(jf)
        fs = pars['sampleRate']
        fs_data = pars['par_sr']
        
    with open(f'./table/staccato_f/note{note}-{seq}.json') as jf2:
        pars2 = json.load(jf2)
    
    # with open(f'./table/vibration_table.json') as jf3:
    #     vib_table = json.load(jf3)
        
        
    pars2['overlapLen'] = 2000
    # note_len = pars['ori_sec']
    note_len = full_note_len
    length = round(expr[6][-1] * note_len * fs) - round(expr[6][0] * note_len * fs)
    pars_len = round(pars['sampleRate'] * pars['ori_sec'])
    start = round(expr[6][0] * note_len * fs) + pars2['attackLen'] - pars2['overlapLen']
    end = round(expr[6][-1] * note_len * fs)
    length_sus = end - start
    
    t = np.arange(length) / fs
    t_sus = t[(pars2['attackLen']-pars2['overlapLen']):]
    
    # pitch and vibration curve
    pitch = linearReSample2(pitch, length)
    b, a = iirfilter(4, Wn=20, fs=fs, ftype="butter", btype="lowpass")
    pitch = timesfilt(b, a, pitch, 8)
    pitch, vib = poly_fit(pitch, d=d_vib)
    vib_range, vib_freq, vfm = vib_pars(vib, fs)
    vib_range, _ = poly_fit(vib_range, d=d_range)
    # vib_freq, _ = poly_fit(vib_freq, d=d_freq)
    vib_freq[:] = np.mean(vib_freq)
    vib_range = np.where(vib_range < 0, 0, vib_range)
    vib_range = vib_range[(pars2['attackLen']-pars2['overlapLen']):]
    vib_freq = vib_freq[(pars2['attackLen']-pars2['overlapLen']):]
    vib = vib[(pars2['attackLen']-pars2['overlapLen']):] * 2
    pitch_sus = pitch[(pars2['attackLen']-pars2['overlapLen']):]
    
    intensity = linearReSample2(np.array(expr[0]), length)
    intensity_sus = intensity[(pars2['attackLen']-pars2['overlapLen']):]
    density = linearReSample2(np.array(expr[2]), length)
    density_sus = density[(pars2['attackLen']-pars2['overlapLen']):]
    hue = linearReSample2(np.array(expr[3]), length)
    hue_sus = hue[(pars2['attackLen']-pars2['overlapLen']):]
    saturation = linearReSample2(np.array(expr[4]), length)
    saturation_sus = saturation[(pars2['attackLen']-pars2['overlapLen']):]
    value = linearReSample2(np.array(expr[5]), length)
    value_sus = value[(pars2['attackLen']-pars2['overlapLen']):]
    
    # hue modify bow position
    hue_sus = np.clip(hue_sus, 0, 135)
    bow_pos = 1 / (hue_sus / 135 * 10 + 2)
    
    base_freq = 440 * np.power(2, (note-69)/12)
    length_sus = length - pars2['attackLen'] + pars2['overlapLen']
    
    if (length_sus <= 0):
        return
    
    mixtone = np.zeros(shape=(length,))
    
    noise1, _ = librosa.load('./colored_noise.wav', sr=fs * (2000 / pars['coloredCutoff1']))
    noise2, _ = librosa.load('./colored_noise.wav', sr=fs * (2000 / pars['coloredCutoff2']))
    
    for partial in range(1, pars['partialAmount']+1):
        over_freq = base_freq * partial
        
        noise_fac = 1.0 + (-1) * density_sus * 0.5
        alpha_fac = 1.0
        mag_fac = 1.0
        
        # attack part
        if quick_atk and partial < pars2['partialAmount']:
            aa = np.array(pars2['alphaAttack'][partial-1])
            ma = np.array(pars2['magAttack'][partial-1])
        else:
            aa = np.array(pars['alphaAttack'][partial-1])
            ma = np.array(pars['magAttack'][partial-1])
            
        
        ag = np.array(pars['alphaGlobal'][partial-1])
        alpha_global = linearReSample2(ag, pars_len)
        alpha_global = alpha_global[start:end]
        pars['alphaLocal']['env'][partial-1][0][0] = (pars['alphaLocal']['env'][partial-1][0][1] + pars['alphaLocal']['env'][partial-1][0][2]) / 2
        pars['alphaLocal']['env'][partial-1][1][0] = (pars['alphaLocal']['env'][partial-1][1][1] + pars['alphaLocal']['env'][partial-1][1][2]) / 2
        env1 = linearReSample2(np.array(pars['alphaLocal']['env'][partial-1][0]), pars_len)
        env2 = linearReSample2(np.array(pars['alphaLocal']['env'][partial-1][1]), pars_len)
        env1 = env1[start:end]
        env2 = env2[start:end]
        c1 = pars['alphaLocal']['spreadingCenter'][partial-1][0]
        c2 = pars['alphaLocal']['spreadingCenter'][partial-1][1]
        fac1 = pars['alphaLocal']['spreadingFactor'][partial-1][0]
        fac2 = pars['alphaLocal']['spreadingFactor'][partial-1][1]
        ng1 = pars['alphaLocal']['noiseGain'][partial-1][0]
        ng2 = pars['alphaLocal']['noiseGain'][partial-1][1]
        noise_start = round(np.random.rand() * 1e+5)
        phase1 = noise1[noise_start:(noise_start+length_sus)]
        noise_start = round(np.random.rand() * 1e+5)
        phase2 = noise2[noise_start:(noise_start+length_sus)]
        alpha_local = np.sin(2*np.pi*c1*t_sus + fac1 * phase1) * env1 * ng1 + np.sin(2*np.pi*c2*t_sus + fac2 * phase2) * env2 * ng2
        alpha_sus = alpha_global + pars['alphaLocal']['gain'][partial-1] * alpha_local * noise_fac * alpha_fac
        aa -= (np.mean(aa[-pars2['overlapLen']:]) - np.mean(alpha_sus[:pars2['overlapLen']]))
        
        mg = np.array(pars['magGlobal'][partial-1])
        mag_global = linearReSample2(mg, pars_len)
        mag_global = mag_global[start:end]
        envelope = linearReSample2(np.array(pars['totalEnv']), pars_len)
        envelope = envelope[start:end]
        envelope = intensity_sus * 0.3 + envelope * 0.02
        mag_ratio = linearReSample2(np.array(pars['magRatio'][partial-1]), pars_len)
        mag_ratio = mag_ratio[start:end]
        mag_global = envelope * mag_ratio
        pars['magLocal']['env'][partial-1][0][0] = (pars['magLocal']['env'][partial-1][0][1] + pars['magLocal']['env'][partial-1][0][2]) / 2
        pars['magLocal']['env'][partial-1][1][0] = (pars['magLocal']['env'][partial-1][1][1] + pars['magLocal']['env'][partial-1][1][2]) / 2
        env1 = linearReSample2(np.array(pars['magLocal']['env'][partial-1][0]), pars_len)
        env2 = linearReSample2(np.array(pars['magLocal']['env'][partial-1][1]), pars_len)
        env1 = env1[start:end]
        env2 = env2[start:end]
        c1 = pars['magLocal']['spreadingCenter'][partial-1][0]
        c2 = pars['magLocal']['spreadingCenter'][partial-1][1]
        fac1 = pars['magLocal']['spreadingFactor'][partial-1][0]
        fac2 = pars['magLocal']['spreadingFactor'][partial-1][1]
        ng1 = pars['magLocal']['noiseGain'][partial-1][0]
        ng2 = pars['magLocal']['noiseGain'][partial-1][1]
        noise_start = round(np.random.rand() * 5e+5)
        phase1 = noise1[noise_start:(noise_start+length_sus)]
        noise_start = round(np.random.rand() * 5e+5)
        phase2 = noise2[noise_start:(noise_start+length_sus)]
        mag_local = np.sin(2*np.pi*c1*t_sus + fac1 * phase1) * env1 * ng1 + np.sin(2*np.pi*c2*t_sus + fac2 * phase2) * env2 * ng2
        
        # most nearing bridge bow position
        if partial % 5 == 0:
            mag_global *= timbre_gain

        # bow position to timbre
        mag_global *= (1 - (1 - np.clip(np.abs(partial - (1 / bow_pos)), 0 ,1)) * (timbre_gain-1))
        
        # saturation
        if over_freq <= 1000:
            mag_global *= np.power(10 ,(-4 + (saturation_sus * 6)) / 20)
            
        # value
        if over_freq >= 3000:
            mag_global *= np.power(10 ,(-4 + (value_sus * 6)) / 20)

        mag_sus = mag_global + pars['magLocal']['gain'][partial-1] * mag_local * noise_fac * mag_fac
        
        # attack AMP
        if partial == 1:
            ma_amp = np.mean(mag_global[:pars2['overlapLen']]) / np.mean(ma[-pars2['overlapLen']:]) * atk_peak
        ma *= ma_amp
        
        # vibration
        if vibration_alpha:
            alpha_sus += np.cumsum(vib) / fs * 2 * np.pi * partial
            
        if vibration_mag and partial < 35:
            shift = round(fs / vfm * vib_phase[partial-1])
            mag_vib = np.pad(vib * vib_amp[partial-1], (shift, 0), mode='edge')
            mag_sus += mag_vib[:length_sus] * mag_global * 0.3
            
        alpha = concat(aa, alpha_sus, pars2['overlapLen'])
        mag = concat(ma, mag_sus, pars2['overlapLen'])

        # global pitch
        alpha += np.cumsum(pitch - base_freq) / fs * 2 * np.pi * partial
        
        # release
        decay = release_decay(over_freq)
        mag[-release_len:] *= np.power(decay, np.arange(release_len))
        
        # partial addition
        tone = np.sin(2 * np.pi * over_freq * t + alpha) * mag
        mixtone += tone
        
    sf.write(des, mixtone, fs)
    print('note witren!!')


    

