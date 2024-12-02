infrastructure and experiments for finding planet nine

plan/log:

- 28/11/2024: sedna detection via shift-stack in the fourier basis, in python using numpy
- 29/11/2024: detection via fft flaky :( so use wavelet 
- 30/11/2024: get wavelet to detect slow-moving nonlinear sources in super low SNR simulated environment with uneven sampling
- 30/11/2024: run on real data, detect simulated source. try large-scale Sedna detection. Failed (although I believe this is due to a combination of large changing artifacts, and the fact that I did not consider that many timesteps in my analysis). Simulating a faint, moving PSF in small skycell blocks are detected.
- 1/12/2024: experiment with different wavelet types and adaptive thresholding. now we detect motion but also a ton of spurious points, so hard to verify if sedna actually detected
- 2/12/2024: TODO: background/sky/stellar-psf sub. why? because it'll probably solve the spurious detection problem. first-pass in numpy then port to cuda and run on 4090s.
- some date in the near future: find planet
