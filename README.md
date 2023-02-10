## Conditional sound effects generation with regularized WGAN

# Abstract 
Over recent years generative models utilizing deep neu- ral networks have demonstrated outstanding capacity in synthesizing high-quality and plausible human speech and music. The majority of research in neural audio synthesis (NAS) targets speech or music, whereas general sound ef- fects such as environmental sounds or Foley sounds have received less attention. In this work, we study the genera- tive performance of NAS models for sound effects with a conditional Wasserstein GAN (WGAN) model. We train our models conditioned on different classes of sound ef- fects and report on their performances in terms of qual- ity and diversity. Many existing GAN models use magni- tude spectrograms which require audio reconstruction us- ing phase estimation after training. The often imperfect reconstruction of the audio signal has led us to propose an additional audio reconstruction loss term for the gen- erator. We show that this additional loss term improves the quality of the audio generation considerably with small sacrifice to the diversity. The results indicate that a con- ditional WGAN model trained on log-magnitude spectro- grams paired with an appropriately weighted reconstruc- tion loss is capable of synthesizing highly plausible sound effects.

Our Baseline model:
<img width="500" alt="Screenshot 2023-02-08 at 21 12 43" src="https://user-images.githubusercontent.com/50271800/217983037-8f257f89-88cd-4491-9f0c-94594f082e4c.png">

Our proposed method:
<img width="500" alt="Screenshot 2023-02-08 at 21 12 32" src="https://user-images.githubusercontent.com/50271800/217983020-bdee347c-8651-406b-bfe1-043247afdc61.png">
