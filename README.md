'Code/autovc' folder is from:
https://github.com/auspicious3000/autovc
(Can find the missing 'Code/Models/AutoVC_Model.ckpt' under a different name in this repo.)

'Code/hifi_gan' folder is from:
https://github.com/jik876/hifi-gan

YouTube video link:
https://www.youtube.com/watch?v=fe7cHEdGSFM

GitHub repository link:
https://github.com/mich-lee/ML-EE475-Project-Repo

Jupyter notebook is found under 'Jupyter/' in that repo.

IMPORTANT NOTE:
In VC_Utils.get_generator_input(...), the sourceSpeakerEmbeddingsHistoryLen option has an effect (something I missed earlier).  Setting it to a higher number is better.  It seems like the "ablation study" I did was not entirely correct in its conclusion: the speaker embedding input to the speech content encoder DOES matter.