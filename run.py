from absl import app, flags
from absl.flags import FLAGS
from src.traj_recovery.trust import *

logging.basicConfig(
    level=logging.INFO,
    filename='src/log/recovery.log',
    filemode='a',
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

flags.DEFINE_integer('traj_len', 5, 'trajectory length in ground truth')
flags.DEFINE_integer('traj_num', 50, 'trajectory num (cars num) of a certain length')
flags.DEFINE_integer('video_time', 8, 'minutes of the output video')
flags.DEFINE_integer('fps', 5, 'frame rate of the input(down sample) video')
flags.DEFINE_integer('node_num', 220, 'how many intersections/videos')
flags.DEFINE_integer('k', 100, 'size of top-k')
flags.DEFINE_float('delta1', 0.6, 'threshold for proximity graph')
flags.DEFINE_float('delta2', 0.6, 'threshold for path selection')
flags.DEFINE_float('lamda', 0.2, 'parameter in path score function')


def main(_argv):
    coarse_heap_no_refine = CoarseHeapNoRefine(FLAGS.traj_len, FLAGS.node_num, FLAGS.video_time, FLAGS.fps, FLAGS.k, FLAGS.delta1, FLAGS.delta2, FLAGS.lamda)
    coarse_heap_no_refine.process_query()


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
