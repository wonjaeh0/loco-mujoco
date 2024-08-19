from experiment_launcher import Launcher
from experiment_launcher.utils import is_local


if __name__ == '__main__':
    LOCAL = is_local()
    TEST = False
    USE_CUDA = False

    N_SEEDS = 1

    # from utils import get_agent
    # from loco_mujoco import LocoEnv
    # env_id="HumanoidMuscle.walk.real"
    # mdp = LocoEnv.make(env_id)
    # agent = get_agent(env_id, mdp, use_cuda=False, sw=None)
    # import pdb;pdb.set_trace()

    launcher = Launcher(exp_name='loco_mujoco_evalution',
                        exp_file='experiment',
                        n_seeds=N_SEEDS,
                        n_cores=1,  # only used for slurm
                        memory_per_core=1500,   # only used for slurm
                        n_exps_in_parallel=10,  # should not be used in slurm
                        days=2,     # only used for slurm
                        hours=0,    # only used for slurm
                        minutes=0,  # only used for slurm
                        use_timestamp=True,
                        )

    default_params = dict(n_epochs=600,
                          n_steps_per_epoch=100000,
                          n_epochs_save=25,
                          n_eval_episodes=10,
                          n_steps_per_fit=1000,
                          use_cuda=USE_CUDA)

    # env_ids = ["Atlas.walk", "Atlas.carry",
    #            "Talos.walk", "Talos.carry",
    #            "UnitreeH1.walk", "UnitreeH1.run", "UnitreeH1.carry",
    #            "UnitreeG1.walk", "UnitreeG1.run",
    #            "HumanoidTorque.walk", "HumanoidTorque.run",
    #            "HumanoidMuscle.walk", "HumanoidMuscle.run",
    #            "UnitreeA1.simple", "UnitreeA1.hard"]
    # env_ids = ["HumanoidMuscle.walk.real", "HumanoidMuscle.run.real",
    #            "HumanoidMuscleExo.walk.real", "HumanoidMuscleExo.run.real"]
    env_ids = ["HumanoidTorque.walk.real", "HumanoidTorque.run.real"]

    for env_id in env_ids:
        launcher.add_experiment(env_id__=env_id, **default_params)

    launcher.run(LOCAL, TEST)
