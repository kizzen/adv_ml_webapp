# A Visual Exploration of Adversarial Attacks

Machine learning models can be easily fooled by applying a small amount of noise to their inputs. This phenomenon is called adversarial machine learning. 

I have developed a Flask web-app which visually explores adversarial attacks: http://kizzen.io/. I trained 2 CNNs - one on MNIST and the other on CIFAR. I then implemented 3 different attacks: FGSM, JSMA and DeepFool. The image on the left is the original image, including it's true classification and what the CNN predicts. The image on the right is the adversarial example, including the amount of pixelated noise/perturbation that was added, and the prediction made using that same model.

To use this tool, select the Dataset (MNIST or CIFAR), the attack (FGSM, JSMA or DeepFool) or let the computer make a random selection, then click "Attack!"

![alt text](https://raw.githubusercontent.com/kizzen/adv_ml_webapp/master/advml_capture.png)

Other projects on my data science portfolio: https://kizzen.github.io/  

### How to Run the App

1. Clone the repository
- `git clone https://github.com/kizzen/adv_ml_webapp.git`
2. In the directory you just cloned, create and activate a new virtual environment
- `cd adv_ml_webapp`
- `conda create -n adv_ml_webapp_env`
- `conda activate adv_ml_webapp_env`
3. Install the necessary packages from requirements.txt into the newly create environment
- `pip install -r requirements.txt`
4. In the `flask/` folder, launch the app from the terminal
- `cd flask`
- `python python advml_app.py`
5. Click the given URL to access the app.: http://127.0.0.1:80.
