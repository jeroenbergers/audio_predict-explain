# audio_predict-explain
This tool classifies if an audio file is synthetic and gives an explanation based on SHAP, Grad-Cam, and Lime for each neural network used for the classification. The tool can be used in an easy-to-use Jupyter notebook. For now, the tool only works on the first 6 seconds of an audio file. 

# What is behind it:

The classification happens based on the resnet based end to end speech detection model, both 1d and 2d are used.
G. Hua, A. B. J. Teoh, and H. Zhang, “Towards end-to-end synthetic speech detection,” IEEE Signal Processing Letters, vol. 28, pp. 1265–1269, 2021. arXiv | IEEE Xplore

The final neural network used is an Integrated Spectro-Temporal Graph Attention Network, created by:
Jung, J. W., Heo, H. S., Tak, H., Shim, H. J., Chung, J. S., Lee, B. J., ... & Evans, N. (2021). AASIST: Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks. arXiv preprint arXiv:2110.01200.

The explanations happen based on an approximation of shap values:
Lundberg, Scott M., and Su-In Lee. "A unified approach to interpreting model predictions." Advances in neural information processing systems 30 (2017).

Gradient-weighted Class Activation Mapping (Grad-CAM):
Selvaraju, Ramprasaath R., et al. "Grad-cam: Visual explanations from deep networks via gradient-based localization." Proceedings of the IEEE international conference on computer vision. 2017.

And LIME:
Ribeiro, M. T., Singh, S., & Guestrin, C. (2016, August). " Why should i trust you?" Explaining the predictions of any classifier. In Proceedings of the 22nd ACM SIGKDD international conference on knowledge discovery and data mining (pp. 1135-1144).

Captum is used for most implementations (https://captum.ai/). 

The given trained models are based on a random sample of files from ASVSpoof2019, ASVspoof 2021, and Wavefake.


# Getting started:
It works on both CPU and GPU, however, the AASIST algorithm requires >10GB ram on CPU and GPU for explainability. As a result, a GPU with >12GB RAM is recommended 
- Pip install -r requirements.txt
- Open dashboard in Jupyter 
- Upload audio sample 
- Predict and explain

# What do I see:
The explanation is in favor of the predicted class. 
Further much is unknown about the predictions and the explanations. 





