
import cv2
from school import Arguments, FrameProcessor, draw_detection

path_to_fd = r"Local\model\face-detection-retail-0004\FP16-INT8\face-detection-retail-0004.xml"
path_to_lm = r"Local\model\landmarks-regression-retail-0009\FP16-INT8\landmarks-regression-retail-0009.xml"
path_to_reid = r"Local\model\face-reidentification-retail-0095\FP16-INT8\face-reidentification-retail-0095.xml"
path_to_gallery = r"Local\gallery"

args = Arguments(Input="0", m_fd=path_to_fd, m_reid=path_to_reid, m_lm=path_to_lm, fg=path_to_gallery, verbose=True,
                 allow_grow=True)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cv2.ocl.setUseOpenCL(False)
frame_processor = FrameProcessor(args)
while True:
    a, frame = cap.read()
    frame = cv2.flip(frame, 1)
    results = frame_processor.process(frame)
    if results is not None:
        frame = draw_detection(frame, frame_processor, results)
    if not args.no_show:
        cv2.imshow('Demo', frame)
        key = cv2.waitKey(1)
        # Quit
        if key in {ord('q'), ord('Q'), 27}:
            break

cap.release()
cv2.destroyAllWindows()