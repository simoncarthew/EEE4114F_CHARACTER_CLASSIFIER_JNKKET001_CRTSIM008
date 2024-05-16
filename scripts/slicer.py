import cv2
import numpy as np

def center_digit(tile):
    # Convert to grayscale and threshold
    gray_tile = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)
    _, thresh_tile = cv2.threshold(gray_tile, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Find contours
    contours, _ = cv2.findContours(thresh_tile, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        # Return a black image if no contours are found
        return np.zeros((28, 28), dtype=np.uint8)

    # Find the bounding box of the largest contour
    contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(contour)
    
    # Extract the digit
    digit = gray_tile[y:y+h, x:x+w]

    # Calculate the scaling factor to fit the digit into 28x28
    scale = min(18.0 / w, 18.0 / h)
    
    # Resize the digit to fit within 28x28
    digit_resized = cv2.resize(digit, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    
    # Create a black canvas of size 28x28
    canvas = np.zeros((28, 28), dtype=np.uint8)
    canvas[:] = 255
    
    # Compute the center offset
    offset_x = (28 - digit_resized.shape[1]) // 2
    offset_y = (28 - digit_resized.shape[0]) // 2
    
    # Place the digit in the center of the canvas
    canvas[offset_y:offset_y + digit_resized.shape[0], offset_x:offset_x + digit_resized.shape[1]] = digit_resized
    
    return canvas

def makeAnExample(pathFrom, pathTo, title):
    img = cv2.imread(pathFrom)
    
    # Define tile dimensions and number of tiles
    tile_height = img.shape[0] // 6
    tile_width = img.shape[1] // 5

    # Iterate through each tile row
    for r in range(6):
        # Iterate through each tile column
        for c in range(5):
            # Calculate the starting and ending pixel coordinates for the tile
            start_r = r * tile_height
            end_r = (r + 1) * tile_height
            start_c = c * tile_width
            end_c = (c + 1) * tile_width
            # Extract the tile from the image
            tile = img[start_r:end_r, start_c:end_c, :]

            # Center the digit in the tile
            centered_tile = center_digit(tile)

            # Invert the image to make the digit white and background black
            inverted = 255 - centered_tile
            # Save the tile as an individual image
            cv2.imwrite(f"{pathTo}/{title}_{r}_{c}.jpg", inverted)

path = "data/hand_crafted/"

makeAnExample("data/RawData/IMG_0698.jpg", path + "0", "Zeros")
print("Finished 0")
makeAnExample("data/RawData/IMG_0699.jpg", path + "1", "Ones")
print("Finished 1")
makeAnExample("data/RawData/IMG_0700.jpg", path + "2", "Twos")
print("Finished 2")
makeAnExample("data/RawData/IMG_0701.jpg", path + "3", "Threes")
print("Finished 3")
makeAnExample("data/RawData/IMG_0702.jpg", path + "4", "Fours")
print("Finished 4")
makeAnExample("data/RawData/IMG_0703.jpg", path + "5", "Fives")
print("Finished 5")
makeAnExample("data/RawData/IMG_0704.jpg", path + "6", "Sixes")
print("Finished 6")
makeAnExample("data/RawData/IMG_0705.jpg", path + "7", "Sevens")
print("Finished 7")
makeAnExample("data/RawData/IMG_0706.jpg", path + "8", "Eights")
print("Finished 8")
makeAnExample("data/RawData/IMG_0707.jpg", path + "9", "Nines")
print("Finished 9")

print("Done")
