# DQN (Deep Q Learning)

Training machines to play CarRacing 2d from OpenAI GYM by implementing Deep Q Learning/Deep Q Network(DQN) with TensorFlow and Keras as the backend.

### Training Results
We can see that the scores(time frames elapsed) stop rising after around 500 episodes as well as the rewards. Thus let's terminate the training and evaluate the model using the last three saved weight files.
<br>
<img src="Output/Result4.png" width="600px">
<br>
<img src="Output/Result1.gif" width="300px">
<br>
<img src="Output/Result2.gif" width="300px">
<br>
<img src="Output/Result3.gif" width="300px">

![trial_500](https://github.com/BadakalaYashwanth/AI-NavQuest-Reinforced-Autonomous-Exploration/assets/170221536/74988ab2-a13c-4a10-80f9-4541f0777e26)


## File Structure

- `train_model.py` The training program.
- `CommonFunctions.py` Some functions that will be used in multiple programs will be put in here.
- `CarRacingDQNAgent.py` The core DQN class. Anything related to the model is placed in here.
- `model.py` The program for playing CarRacing by the model.
- `keyboardModel.py` The program for playing CarRacing with the keyboard.
- `save/` The default folder to save the trained model.


