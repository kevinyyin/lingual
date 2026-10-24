# corpus/noise — background noise beds for the `noisy` category

Drop mono or stereo `.wav` files of real café / street ambience here (for example CC0
recordings from freesound.org). `bench prepare` mixes one under each `noisy` clip at
`audio.noise_snr_db` (default 10 dB). Files are chosen deterministically per clip.

If this folder has no wav files, `prepare` falls back to **babble noise**: six unrelated
corpus talkers summed at random offsets, a standard stand-in for café chatter. The noise
source used for every clip is recorded in `corpus/prepared/index.json` (`meta.noise_source`).
