import gym
import glfw

is_pressed_left  = False # control left
is_pressed_right = False # control right
is_pressed_space = False # control gas
is_pressed_shift = False # control break
is_pressed_esc   = False # exit the game
steering_wheel = 0 # init to 0
gas            = 0 # init to 0
break_system   = 0 # init to 0

def key_callback(window, key, scancode, action, mods):
    global is_pressed_left
    global is_pressed_right
    global is_pressed_space
    global is_pressed_shift
    global is_pressed_esc

    if key == glfw.KEY_LEFT:
        is_pressed_left = (action == glfw.PRESS)
    elif key == glfw.KEY_RIGHT:
        is_pressed_right = (action == glfw.PRESS)
    elif key == glfw.KEY_SPACE:
        is_pressed_space = (action == glfw.PRESS)
    elif key == glfw.KEY_LEFT_SHIFT:
        is_pressed_shift = (action == glfw.PRESS)
    elif key == glfw.KEY_ESCAPE:
        is_pressed_esc = (action == glfw.PRESS)

def update_action():
    global steering_wheel
    global gas
    global break_system

    if is_pressed_left ^ is_pressed_right:
        if is_pressed_left:
            if steering_wheel > -1:
                steering_wheel -= 0.1
            else:
                steering_wheel = -1
        if is_pressed_right:
            if steering_wheel < 1:
                steering_wheel += 0.1
            else:
                steering_wheel = 1
    else:
        if abs(steering_wheel - 0) < 0.1:
            steering_wheel = 0
        elif steering_wheel > 0:
            steering_wheel -= 0.1
        elif steering_wheel < 0:
            steering_wheel += 0.1
    if is_pressed_space:
        if gas < 1:
            gas += 0.1
        else:
            gas = 1
    else:
        if gas > 0:
            gas -= 0.1
        else:
            gas = 0
    if is_pressed_shift:
        if break_system < 1:
            break_system += 0.1
        else:
            break_system = 1
    else:
        if break_system > 0:
            break_system -= 0.1
        else:
            break_system = 0

if __name__ == '__main__':
    env = gym.make('CarRacing-v0')
    env.reset()

    glfw.init()
    window = glfw.create_window(640, 480, "Car Racing", None, None)
    glfw.make_context_current(window)
    glfw.set_key_callback(window, key_callback)

    counter = 0
    total_reward = 0
    while not is_pressed_esc:
        env.render()
        update_action()
        action = [steering_wheel, gas, break_system]
        state, reward, done, info = env.step(action)
        counter += 1
        total_reward += reward
        print('Action:[{:+.1f}, {:+.1f}, {:+.1f}] Reward: {:.3f}'.format(action[0], action[1], action[2], reward))
        if done:
            print("Restart game after {} timesteps. Total Reward: {}".format(counter, total_reward))
            counter = 0
            total_reward = 0
            state = env.reset()
            continue

    env.close()
    glfw.terminate()
