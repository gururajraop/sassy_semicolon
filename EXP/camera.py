def iterate_over_camera():
    import cv2
    cap = cv2.VideoCapture(-1)

    idx = 0
    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()
        if frame is None:
            continue
        yield 0, idx, np.asarray(frame)
        idx += 1




cv2.imshow('image',img)
cv2.waitKey(0)
