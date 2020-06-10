import cv2


def templateMatch(originalImagePath, templatePath):
    # Read original and template image
    original_image = cv2.imread(originalImagePath)
    templateOri = cv2.imread(templatePath)

    # Convert to grayscale
    imageGray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    templateGray = cv2.cvtColor(templateOri, cv2.COLOR_BGR2GRAY)

    # assign width and height of template in w and h
    h, w = templateGray.shape
    # Perform match operations.
    res = cv2.matchTemplate(imageGray, templateGray, cv2.TM_CCOEFF_NORMED)

    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    cv2.rectangle(original_image, top_left, bottom_right, (0, 0, 0), 4)

    cv2.imshow('Template', templateOri)
    cv2.imshow('Detected Template', original_image)
    cv2.waitKey(0)


def main():
    originalImagePath, templatePath = 'King_test_img.png', 'Jack.png'
    templateMatch(originalImagePath, templatePath)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()