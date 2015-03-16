usage = """a module that handles all dft calculations """

import numpy as np

#=================================================
# windowing functions
#=================================================
###
def window(vec, kind="kaiser", **kwargs):
        """
        compute the window function for vec
        """
        N = len(vec)
        if kind=='tukey':
                if not kwargs.has_key('alpha'):
                        raise ValueError, "must supply alpha as optional argument with kind=tukey"
                return tukey(N, kwargs['alpha'])
        elif kind=="kaiser":
                if not kwargs.has_key('beta'):
                        raise ValueError, "must supply beta as optional argument with kind=kaiser"
                return np.kaiser(N, kwargs['beta'])
        elif kind=='hamming':
                return np.hamming(N)
        elif kind=='hanning':
                return np.hanning(N)
        elif kind=='bartlett':
                return np.bartlett(N)
        elif kind=='blackman':
                return np.blackman(N)
        else:
                raise ValueError, "kind=%s not understood"%kind

###
def tukey(N, alpha):
        """
        generate a tukey window

        The Tukey window, also known as the tapered cosine window, can be regarded as a cosine lobe of width \alpha * N / 2
                that is convolved with a rectangle window of width (1 - \alpha / 2). At \alpha = 1 it becomes rectangular, and
                at \alpha = 0 it becomes a Hann window.
 
        """
        # Special cases
        if alpha <= 0:
                return np.ones(N) #rectangular window
        elif alpha >= 1:
                return np.hanning(N)

        # Normal case
        x = np.linspace(0, 1, N)
        w = np.ones(x.shape)

        # first condition 0 <= x < alpha/2
        first_condition = x<alpha/2
        w[first_condition] = 0.5 * (1 + np.cos(2*np.pi/alpha * (x[first_condition] - alpha/2) ))

        # second condition already taken care of

        # third condition 1 - alpha / 2 <= x <= 1
        third_condition = x>=(1 - alpha/2)
        w[third_condition] = 0.5 * (1 + np.cos(2*np.pi/alpha * (x[third_condition] - 1 + alpha/2)))

        return w

#=================================================
# DFT utilities
#=================================================
###
def dft(vec, dt=1.0):
        """
        computes the DFT of vec
        returns the one-sides spectrum
        """
        N = len(vec)

        dft_vec = np.fft.fft(vec)*dt
        freqs = np.fft.fftfreq(N, d=dt)

        freqs = np.fft.fftshift(freqs)
        truth = freqs>=0

        return np.fft.fftshift(dft_vec)[truth], freqs[truth]

###
def idft(dft_vec, dt=1.0):
        """
        computes the inverse DFT of vec
        takes in the one-sided spectrum
        """
        N = len(dft_vec) ### if N is even, then n is even
                         ### if N is odd, then n is odd

        if N%2: ### if N is odd, n is odd
                n = 2*N-1
        else: ### if N is even, n is even
                n = 2*N

        seglen = n*dt ### length of time series

        vec = np.empty((n,), complex)
        vec[:N] = dft_vec
        if n%2: ### odd number of points
                vec[N:] = np.conjugate(dft_vec[1:])[::-1]
        else: ### even number of points
                vec[N:] = np.conjugate(dft_vec)[::-1]

        vec = np.fft.ifft( vec ) / seglen
        time = np.arange(0, seglen, dt)

        return vec, time

#=================================================
# resampling utilities
#=================================================
###
def resample(vec, dt, new_dt, method="average"):
        """
        vec = time-series array
        dt = current time spacing
        new_dt = desired time spacing
        """
        if method == "interpolate":
                return __resample_interp(vec, dt, new_dt)

        elif method == "average":
                return __resample_average(vec, dt, new_dt)

        elif method == "lowpass":
                return __resample_lowpass(vec, dt, new_dt)

        else:
                raise ValueError, "method=%s not understood"%method

###
def __resample_interp(vec, dt, new_dt):
        """
        resample by linear interpolation. This is known to be a bad idea
        """
        N = len(vec)
        return np.interp( np.arange(o, N*dt, new_dt), np.arange(0, N*dt, dt), vec), new_dt

###
def __resample_average(vec, dt, new_dt):
        """
        averages neighbouring points to downsample the time series
        if len(vec) is odd, we pad with a zero.
        """
        if dt > new_dt:
                raise ValueError, "cannot average samples to a higher sampling rate"

        N = len(vec)
        if N%2:
                raise ValueError, "averaging to downsample is poorly defined for vectors with odd numbers of elements"
                vec = np.concatenate([vec, np.zeros(1)])
                N += 1

        if dt == new_dt: ### we're done
                return vec, dt
        elif (new_dt/dt)%2: ### bad shape
                raise ValueError, "new_dt/dt must be a power of 2"

        else: ### recursive call to downsample
                new_vec = np.sum(np.reshape(vec, (N/2, 2)), axis=1) * 0.5 ### average neighbouring points
                return __resample_average(new_vec, dt*2, new_dt) ### recursive call with new resampled data

###
def __resample_lowpass(vec, dt, new_dt):
        """
        computes the FFT and passes the signal through a low-pass filter

        CURRENTLY we use a top-hat function at the Nyquist frequency.
        we may want to consider a Butterworth filter of high order (LAL uses n=20), which is a harmonic
        approximation to the top-hat.
        """
        if dt == new_dt:
                return vec, dt
        elif dt > new_dt:
                raise ValueError, "don't know how to upsample data"

        dft_vec, freqs = dft(vec, dt=dt) ### take DFT

        new_nyquist = 0.5/new_dt
        new_dft_vec = dft_vec[freqs<new_nyquist] ### throw out all points above new_nyquist

        return idft(new_dft_vec*new_dt/dt, dt=new_dt)[0], new_dt ### take iDFT and return (only the vector, not the time)

#=================================================
# PSD utilities
#=================================================
###
def estimate_psd(vec, num_segs=1, overlap=0, dt=1.0):
        """
        estimates the PSD using a DFT
        divides vec into "num_segs" with "overlap" shared entries between neighbors
        returns the average PSD from these samples

        WARNING: your logic on how to split segments may be fragile...

        """
        N = len(vec)
        if overlap > N - num_segs:
                raise ValueError, "overlap is too big!"

        n = 1.0*N/num_segs + (num_segs-1.0)/num_segs * overlap ### number of elements per segment

        n = int(n)
        overlap = int(overlap)
        seglen = dt*n

        ### compute dfts for each segment separately
        psds = np.empty((n/2, num_segs), complex)
        for segNo in xrange(num_segs):
                start = segNo*(n-overlap)
                psd, freqs = dft(vec[start:start+n], dt=dt)
                psds[:,segNo], freqs = dft(vec[start:start+n], dt=dt)

        ### average
        mean_psd = np.sum(psds.real**2 + psds.imag**2, axis=1) / (seglen*num_segs)

        return mean_psd, freqs
