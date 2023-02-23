import datetime
from openvino.inference_engine import IECore
from pathlib import Path
from landmarks_detector import LandmarksDetector
from face_detector import FaceDetector
from faces_database import FacesDatabase
from face_identifier import FaceIdentifier
import cv2
from utils import crop


class Arguments:
    def __init__(self, Input, fg, m_fd, m_lm, m_reid, loop=False, output=None, limit=1000,
                 output_resolution=None,
                 no_show=False, crop_size=(0, 0), match_algo='HUNGARIAN', utilization_monitors="", run_detector=False,
                 allow_grow=False, fd_input_size=(0, 0), d_fd="CPU", d_lm="CPU", d_reid='CPU', cpu_lib="",
                 gpu_lib="", verbose=False, perf_stats=False, t_fd=0.99, t_id=0.3, exp_r_fd=1.15):
        self.input = Input
        self.loop = loop
        self.output = output
        self.output_limit = limit
        self.output_resolution = output_resolution
        self.no_show = no_show
        self.crop_size = crop_size
        self.match_algo = match_algo
        self.utilization_monitors = utilization_monitors
        self.fg = fg
        self.run_detector = run_detector
        self.allow_grow = allow_grow
        self.m_fd = Path(m_fd)
        self.m_lm = Path(m_lm)
        self.m_reid = Path(m_reid)
        self.fd_input_size = fd_input_size
        self.d_fd = d_fd
        self.d_lm = d_lm
        self.d_reid = d_reid
        self.cpu_lib = cpu_lib
        self.gpu_lib = gpu_lib
        self.verbose = verbose
        self.perf_stats = perf_stats
        self.t_fd = t_fd
        self.t_id = t_id
        self.exp_r_fd = exp_r_fd


class FrameProcessor:
    QUEUE_SIZE = 16

    def get_config(self, device):
        config = {
            "PERF_COUNT": "YES" if self.perf_count else "NO",
        }
        if device == 'GPU' and self.gpu_ext:
            config['CONFIG_FILE'] = self.gpu_ext
        return config

    def __init__(self, args):
        self.gpu_ext = args.gpu_lib
        self.perf_count = args.perf_stats
        self.allow_grow = args.allow_grow and not args.no_show

        ie = IECore()
        if args.cpu_lib and 'CPU' in {args.d_fd, args.d_lm, args.d_reid}:
            ie.add_extension(args.cpu_lib, 'CPU')

        self.face_detector = FaceDetector(ie, args.m_fd,
                                          args.fd_input_size,
                                          confidence_threshold=args.t_fd,
                                          roi_scale_factor=args.exp_r_fd)
        self.landmarks_detector = LandmarksDetector(ie, args.m_lm)
        self.face_identifier = FaceIdentifier(ie, args.m_reid,
                                              match_threshold=args.t_id,
                                              match_algo=args.match_algo)
        self.face_detector.deploy(args.d_fd, self.get_config(args.d_fd))
        self.landmarks_detector.deploy(args.d_lm, self.get_config(args.d_lm), self.QUEUE_SIZE)
        self.face_identifier.deploy(args.d_reid, self.get_config(args.d_reid), self.QUEUE_SIZE)
        self.faces_database = FacesDatabase(args.fg, self.face_identifier,
                                            self.landmarks_detector,
                                            self.face_detector if args.run_detector else None, args.no_show)
        self.face_identifier.set_faces_database(self.faces_database)

    def process(self, frame, emotion_recog=True):
        orig_image = frame.copy()
        rois = self.face_detector.infer((frame,))
        if rois:
            landmarks = self.landmarks_detector.infer((frame, rois))
            face_identities, unknowns = self.face_identifier.infer((frame, rois, landmarks))
            if self.allow_grow and len(unknowns) > 0:
                for i in unknowns:
                    # This check is preventing asking to save half-images in the boundary of images
                    if rois[i].position[0] == 0.0 or rois[i].position[1] == 0.0 or \
                            (rois[i].position[0] + rois[i].size[0] > orig_image.shape[1]) or \
                            (rois[i].position[1] + rois[i].size[1] > orig_image.shape[0]):
                        continue
                    crop_image = crop(orig_image, rois[i])
                    name = self.faces_database.ask_to_save(crop_image)
                    if name:
                        id = self.faces_database.dump_faces(crop_image, face_identities[i].descriptor, name)
                        face_identities[i].id = id

            return [rois, landmarks, face_identities]
        else:
            return None


def draw_detection(frame, frame_processor, detections):
    size = frame.shape[:2]
    for roi, landmarks, identity in zip(*detections):
        text = frame_processor.face_identifier.get_identity_label(identity.id)
        if identity.id != FaceIdentifier.UNKNOWN_ID:
            text += ' %.2f%%' % (100.0 * (1 - identity.distance))

        xmin = max(int(roi.position[0]), 0)
        ymin = max(int(roi.position[1]), 0)
        xmax = min(int(roi.position[0] + roi.size[0]), size[1])
        ymax = min(int(roi.position[1] + roi.size[1]), size[0])
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 220, 0), 1)
        cv2.putText(frame, text, (xmin, ymax - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 0), 1, cv2.LINE_AA)
    return frame


