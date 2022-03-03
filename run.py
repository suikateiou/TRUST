from absl import app, flags
from absl.flags import FLAGS
from src.traj_recovery.trust import *

logging.basicConfig(
    level=logging.INFO,
    filename='src/log/recovery.log',
    filemode='a',
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

flags.DEFINE_integer('traj_len', 5, 'trajectory length in ground truth')
flags.DEFINE_integer('video_time', 10, 'minutes of the output video')
flags.DEFINE_integer('fps', 10, 'frame rate of the input (down-sampled) video')
flags.DEFINE_integer('node_num', 30, 'how many intersections/videos')
flags.DEFINE_integer('k', 80, 'size of top-k')
flags.DEFINE_float('delta', 0.45, 'threshold for proximity graph and path selection')


def main(_argv):
    coarse_heap_no_refine = CoarseHeapNoRefine(FLAGS.traj_len, FLAGS.node_num, FLAGS.video_time, FLAGS.fps, FLAGS.k, FLAGS.delta)
    coarse_heap_no_refine.process_query()


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
