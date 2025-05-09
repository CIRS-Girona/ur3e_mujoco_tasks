import os
from collections import deque
from stable_baselines3.common.callbacks import BaseCallback
from torch.utils.tensorboard import SummaryWriter

class SuccessTrackerCallback(BaseCallback):
    def __init__(
        self,
        n_episodes=100,
        next_stage_threshold=1.0,
        log_dir="logs_custom",
        save_path="models",
        verbose=0,
    ):
        super(SuccessTrackerCallback, self).__init__(verbose)
        self.n_episodes = n_episodes
        self.next_stage_threshold = next_stage_threshold
        self.success_history = deque(maxlen=n_episodes)
        self.rewards_history = deque(maxlen=n_episodes)
        self.lengths_history = deque(maxlen=n_episodes)
        self.current_rewards = []
        self.current_length = 0
        self.episodes_done = 0
        self.logging_timestep = 0

        self.log_dir = log_dir
        self.save_path = save_path
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(save_path, exist_ok=True)

        self.current_stage = None
        self.writer = None

        self.save_frequency = 50000 # number of timesteps between each saved model

    def _on_training_start(self) -> None:
        # unwrap environment to access env attributes
        env = self.training_env.envs[0]
        while hasattr(env, "env"):
            env = env.env  # unwrap Monitor and any wrappers
        self.env_unwrapped = env

        # set final learning stage
        self.max_learning_stage = getattr(getattr(self.env_unwrapped, "curriculum", None), "final_stage", None)

        # create first logging data
        self.current_stage = getattr(self.env_unwrapped, "learning_stage")
        stage_logdir = os.path.join(self.log_dir, f"stage_{self.current_stage}")
        if os.path.exists(stage_logdir): # in case of continuing training
            stage_logdir = stage_logdir + "_resume"
        self.writer = SummaryWriter(log_dir=stage_logdir)

    def _on_step(self) -> bool:
        self.current_rewards.append(self.locals["rewards"])
        self.current_length += 1

        # check learning stage
        stage = getattr(self.env_unwrapped, "learning_stage", 0)
        # reset all history if stage has changed
        if stage != self.current_stage:
                self.model.save(os.path.join(self.save_path, f"model_stage_{self.current_stage}.zip"))
                self.writer.close()
                self.current_stage = stage
                self.writer = SummaryWriter(log_dir=os.path.join(self.log_dir, f"stage_{stage}"))
                self.success_history.clear()
                self.rewards_history.clear()
                self.lengths_history.clear()
                self.episodes_done = 0
                self.logging_timestep = 0
        
        self.logging_timestep += 1

        # save checkpoint models
        if self.logging_timestep % self.save_frequency == 0:
            checkpoint_filename = f"model_stage_{self.current_stage}_step_{self.logging_timestep}.zip"
            self.model.save(os.path.join(self.save_path, checkpoint_filename))

        if self.locals["dones"]:
            # Get info and success
            info = self.locals["infos"]
            success = False
            if isinstance(info, list) and len(info) > 0:
                success = info[0].get("is_success", False)

            total_reward = sum(self.current_rewards)
            ep_length = self.current_length
            self.current_rewards = []
            self.current_length = 0

            # Update histories
            self.success_history.append(int(success))
            self.rewards_history.append(total_reward)
            self.lengths_history.append(ep_length)
            self.episodes_done += 1

            # Compute stats
            success_rate = sum(self.success_history) / len(self.success_history)
            ep_rew_mean = sum(self.rewards_history) / len(self.rewards_history)
            ep_len_mean = sum(self.lengths_history) / len(self.lengths_history)

            # TensorBoard logging
            self.writer.add_scalar("custom/success_rate", success_rate, self.logging_timestep)
            self.writer.add_scalar("custom/ep_rew_mean", ep_rew_mean, self.logging_timestep)
            self.writer.add_scalar("custom/ep_len_mean", ep_len_mean, self.logging_timestep)

            if self.verbose > 0:
                print("===========================")
                print(f"[Stage {stage}] Ep {self.episodes_done} | Success Rate: {success_rate:.2f}% | Mean Reward: {ep_rew_mean.item():.2f} | Mean Len: {ep_len_mean:.1f}")
                print("===========================")

            # Advance curriculum if needed
            if (success_rate >= self.next_stage_threshold) and (self.episodes_done >= self.n_episodes):
                if hasattr(self.env_unwrapped, "learning_stage"):
                    self.env_unwrapped.learning_stage += 1 if self.env_unwrapped.learning_stage < self.max_learning_stage else 0
                    print(f"Advance to stage {self.env_unwrapped.learning_stage}")
                else:
                    raise AttributeError("Environment must have a 'learning_stage' attribute.")

        return True

    def _on_training_end(self) -> None:
        # save final model
        stage_savefile = os.path.join(self.save_path, f"model_stage_{self.current_stage}_final.zip")
        if os.path.exists(stage_savefile): # in case of continuing training and stage still doesn't advance at all
            stage_savefile = os.path.join(self.save_path, f"model_stage_{self.current_stage}_final2.zip")
        self.model.save(stage_savefile)
        
        # close logger
        if self.writer:
            self.writer.close()
