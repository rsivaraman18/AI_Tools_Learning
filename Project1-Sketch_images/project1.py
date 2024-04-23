import cv2,os

#input_image_path = r"D:\Siva\2.My-Learnings\02_Siva_Learnings\Project1\Trained_Images\Image2.jpg"

input_image_path = "Trained_Images\Image2.jpg"
image_name_with_extension = os.path.basename(input_image_path)  
image_name_without_extension = os.path.splitext(image_name_with_extension)[0]

image = cv2.imread(input_image_path)

count = 1
for value in range(140,190,10):
    
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Invert the grayscale image
    inverted_gray_image = value - gray_image               

    # Apply Gaussian blur to the inverted image
    blurred_image = cv2.GaussianBlur(inverted_gray_image, (21, 21), 0)

    # Invert the blurred image
    inverted_blurred_image = 255 - blurred_image

    # Create the sketch by combining the inverted blurred image with the original grayscale image
    sketch = cv2.divide(gray_image, inverted_blurred_image, scale=256.0)

    # Define the contrast and brightness adjustment parameters
    alpha = 1.5  # Contrast control (1.0-3.0)
    beta = 30    # Brightness control (0-100)
 
    # Apply contrast and brightness adjustment
    dark_sketch = cv2.convertScaleAbs(sketch, alpha=alpha, beta=beta)
    output_image = "{}_sketch{}".format(image_name_without_extension,count)
    count = count + 1
    #output_location = "Sketched_Images/{}.jpg".format(output_image)      # JPG
    output_location = "Sketched_Images/{}.png".format(output_image)        #png
     
    cv2.imwrite(output_location, dark_sketch)
    print(f"Image sketch saved at: {output_location}")
