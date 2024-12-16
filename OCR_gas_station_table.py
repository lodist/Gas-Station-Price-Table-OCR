
def process_image(image_path=None):
    # print("DEBUG: Entered process_image function")
    try:
        if image_path:
            # %%
            from PIL import Image, ImageEnhance
            import os
            import cv2 as cv
            import re
            import ssl
            from tensorflow.lite.python import interpreter as tflite
            import numpy as np
            import shutil
            import sys
            import logging
            logging.basicConfig(level=logging.DEBUG)  
            logger = logging.getLogger('ppocr')
            logger.setLevel(logging.DEBUG)
            from paddleocr import PaddleOCR
            # %%
            if image_path:
                # print(f"Image Path: {image_path}, Type: {type(image_path)}")
                img_original = cv.imread(image_path)
            
                # %%
                # EXTRACT TEXT FROM IMAGE
            
                # Setup model
                ocr_model = PaddleOCR(lang='en')
                result = ocr_model.ocr(image_path)
                # print(f"OCR Result: {result}")  # Debugging line

                # Check if OCR result is empty or None and return if it is
                if not result:
                    print("No text found in image.")

                words = []
                for sublist in result:
                    for res in sublist:
                        word = res[1][0]
                        words.append(word)

                # print(words)
                boxes = []
                for sublist in result:
                    for item in sublist:
                        box = item[0]  # this is the list of four coordinates
                        boxes.append(box)
                # print(boxes)
            
                # Calculate the mean x-coordinate of all boxes
                means = []
                for box in boxes:
                    xs = [coord[0] for coord in box]
                    means.append(np.mean(xs))
                mean_x = np.mean(means)
            
                # Select only the boxes on the left 3/4 of the image
                left_boxes = [box for box in boxes if np.mean([coord[0] for coord in box]) < (mean_x * 1.2)]
            
                # Get the indices of these boxes in the original list
                left_indices = [boxes.index(box) for box in left_boxes]
            
                # Use these indices to select the corresponding words
                left_words = [words[i] for i in left_indices]
                # print(left_words)
            
                avg_ys = []  
                for box in left_boxes:
                    ys = [coord[1] for coord in box]
                    avg_ys.append(np.mean(ys))
            
                # Pair each left box with its corresponding word and average y-coordinate
                pairs = list(zip(left_boxes, left_words, avg_ys))
            
                # Sort the pairs based on the average y-coordinate
                pairs.sort(key=lambda pair: pair[2])
            
                # Initialize a list to hold the final joined words
                final_words = []
            
                # Process each pair
                for i in range(len(pairs)):
                    current_word = pairs[i][1]
                    current_avg_y = pairs[i][2]
                    
                    # Check if this is the first pair, add its word to final_words
                    if i == 0:
                        final_words.append(current_word)
                    else:
                        # Check if the current word contains a numeric value (no spaces) and no letters
                        has_numeric_value = any(char.isdigit() for char in current_word)
                        has_letters = any(char.isalpha() for char in current_word)
                        
                        if has_numeric_value and not has_letters:
                            # Check the last word added to final_words to prevent consecutive numbers
                            last_word = final_words[-1]
                            if not any(char.isdigit() for char in last_word):
                                final_words[-1] += ' ' + current_word
                        else:
                            # Check the last word added to final_words to prevent consecutive words
                            last_word = final_words[-1]
                            if any(char.isdigit() for char in last_word):
                                final_words.append(current_word)
                            else:
                                # Remove the last word if it's a duplicate and add the current word
                                final_words[-1] = current_word

                # Remove duplicates if they occur due to the above process
                for i in range(len(final_words)-2, 0, -1):
                    if final_words[i-1] == final_words[i+1]:
                        final_words[i-1] = final_words[i-1] + ' ' + final_words[i]
                        del final_words[i]

                sorted_definition_list = final_words

            
                if not sorted_definition_list:  # Check if the list is empty
                    result_list = []
                else:
                    result_list = [sorted_definition_list[0]]
                    for i in range(1, len(sorted_definition_list)):
                        current_word = sorted_definition_list[i]
                        previous_word = sorted_definition_list[i - 1]
            
                        if current_word == previous_word and i > 1:
                            # If the same, append the current word to the previous word in result_list
                            result_list[-2] += ' ' + current_word
                        else:
                            result_list.append(current_word)
            
                    sorted_definition_list = result_list
            
                # %%
                #1
            
                #DRAW CONTOURS AROUND RECOGNIZED AREAS
                def are_contours_similar(cnt1, cnt2, similarity_threshold=0.05):
                    return cv.matchShapes(cnt1, cnt2, cv.CONTOURS_MATCH_I1, 0) < similarity_threshold
            
                img_color = img_original
            
                BCOLOR = (75, 0, 130)
                THICKNESS = 4
            
                height = img_color.shape[0]
                min_height = 500
                max_height = 800
                scale_factor = min(max(min_height / height, 1), max_height / height)
                img_color = cv.resize(img_color, None, None, fx=scale_factor, fy=scale_factor)
                img = cv.cvtColor(img_color, cv.COLOR_BGR2GRAY)

                blurred = cv.GaussianBlur(img, (3,3), 0)
                retry_count = 0
                MAX_RETRIES = 8
                block_size = 9
                bilateral_param = 3
                roi_images = []
                while retry_count < MAX_RETRIES:
                    
                    img_temp = img_color.copy()
                    blurred_current = cv.GaussianBlur(img, (3,3), 0) 
                    blurred_current = cv.bilateralFilter(blurred_current, bilateral_param, 75, 75)
            
                    thresh = cv.adaptiveThreshold(blurred_current, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                                cv.THRESH_BINARY_INV, block_size, 1)
            
                    cnts, _ = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
                    cnts = sorted(cnts, key = cv.contourArea, reverse=True)[:10]
            
                    # Check if cnts is empty
                    if not cnts:
                        retry_count += 1
                        block_size = 13
                        bilateral_param += 1 
                        continue

                # #SELECT CONTOURS WITH SIMILAR AREA
                    target_area = cv.contourArea(cnts[0])  # Set the area of the first contour as the target area
                    threshold = 0.3 * target_area # Define a threshold for similarity
                    roi_images = []
            
                    for i, cnt in enumerate(cnts):
                        contour_area = cv.contourArea(cnt)
                        if abs(contour_area - target_area) < threshold:
                            x, y, w, h = cv.boundingRect(cnt) # Calculate the bounding rectangle of the contour
            
                            if contour_area < (img.shape[0] * img.shape[1]) / 20:
                                roi = img.copy() # If the contour is smaller than a tenth of the image, pass the entire image
                            else:
                                roi = img[y:y+h, x:x+w] # Extract the region of interest (ROI) from the original image
            
                            roi_images.append((roi, y))
            
                    roi_images.sort(key=lambda item: item[1]) # Sort the roi_images by the y value of their bounding boxes, in ascending order
                    
                    # print(bilateral_param)
            
                    if len(roi_images) < 2:
                        retry_count += 1
                        block_size = 13
                        bilateral_param += 1
                        continue
            
                    if len(roi_images) >= 2:
                        if are_contours_similar(cnts[0], cnts[1]):
                            retry_count += 1
                            block_size = 13
                            bilateral_param += 1
                            continue
            
                    if len(roi_images) != len(sorted_definition_list):
                        retry_count += 1
                        block_size = 13
                        bilateral_param += 1 
                        continue
                    # If successful, break out of the loop
                    break
            
                # Draw the contours
                cv.drawContours(img_color, cnts, -1, BCOLOR, THICKNESS);
                # cv.imshow("Target Contour", img_color)
                # cv.waitKey(0)
            
                # %%
                # #1
                #SAVE MOST IMPORTANT CONTOURS
            
                # Define a function to enhance the resolution of an image
                def enhance_resolution(image_array, scale_factor):
                    image = Image.fromarray(image_array)    # Convert the image array to a PIL Image    
                    # Calculate the new width and height based on the scale factor
                    width = image.width * scale_factor
                    height = image.height * scale_factor
                    resized_image = image.resize((int(float(width)), int(float(height))), Image.BICUBIC)    # Resize the image using the BICUBIC interpolation method
                    enhanced_image_array = np.array(resized_image)    # Convert the resized image back to a numpy array
                    # cv.imshow("Target Contour", enhanced_image_array)
                    # cv.waitKey(0)
                    return enhanced_image_array    # Return the enhanced image array
            
                # Specify the scale factor for image enhancement
                scale_factor = 3  # Increase resolution by a factor of 3
            
                # Dictionary to store the enhanced images
                enhanced_images_dict = {}
            
                # Loop over the sorted list of ROI images and their y-values
                for i, (roi, _) in enumerate(roi_images):
                    enhanced_image = enhance_resolution(roi, scale_factor)    # Enhance the resolution of the ROI
                    # Store the enhanced image in the dictionary with a unique key
                    enhanced_images_dict[f"contour_i_{i+1}"] = enhanced_image
            
                # %%
                #1
                #CONTAIN EVERY OBJECT IN A RECTANGLE
          
                all_contours_list = []
            
                def process_images(kernel_size):
                    all_segment_dicts = {} 
                    images_saved_per_contour = {}  
            
                    # Loop over the keys and values in the image_dict_pre_2
                    for filename, roi in enhanced_images_dict.items():  # Note: changed roi_color to roi
            
                        # Define the minimum and maximum heights
                        min_height = 800
                        max_height = 1200
            
                        # Calculate the scaling factor based on the desired height range
                        scale_factor = min(max(min_height / height, 1), max_height / height)
            
                        # Resize the image using the calculated scale factor
                        roi = cv.resize(roi, None, None, fx=scale_factor, fy=scale_factor)
                        
                        # Convert the image to grayscale if needed
                        if len(roi.shape) > 2:
                            gray_roi = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
                        else:
                            gray_roi = roi
            
                        # Apply adaptive thresholding
                        _, edged = cv.threshold(
                            gray_roi, 200, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU
                        )
            
                        edged = 255 - edged
            
                        # Apply the kernel
                        kernel = cv.getStructuringElement(cv.MORPH_RECT, kernel_size)
                        dilated = cv.dilate(edged, kernel, iterations=2)
            
                        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (1, 3),)
                        eroded = cv.erode(dilated, kernel, iterations=1)
            
                        # cv.imshow("Eroded", eroded)
                        # cv.waitKey(0)
            
                        h = roi.shape[0]
                        ratio = int(float(h * 0.07))
                        eroded[-ratio:,] = 0
                        eroded[:, :ratio] = 0
            
                        cnts, _ = cv.findContours(eroded, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
                        
                        canvas = eroded.copy()
                        digits_cnts = []
                        for cnt in cnts:
                            (x, y, w, h) = cv.boundingRect(cnt)
                            if h > 20:
                                digits_cnts += [cnt]
                                cv.drawContours(canvas, [cnt], -1, (75, 0, 130), thickness=2)
                                cv.rectangle(canvas, (x, y), (x + w, y + h), (75, 0, 130), thickness=2)
                    
                        # Calculate the height and width of the image
                        img_height, img_width = roi.shape[:2]
                        
                        # Thresholds for the contours
                        height_threshold = 0.85 * img_height
                        width_threshold = 0.85 * img_width
                        
                        # Calculate the threshold area for the largest contour to be removed
                        threshold_area = 0.1 * img_height * img_width
            
                        # Remove contours that are too large
                        digits_cnts = [cnt for cnt in digits_cnts 
                                    if cv.boundingRect(cnt)[3] <= height_threshold 
                                    and cv.boundingRect(cnt)[2] <= width_threshold 
                                    and cv.contourArea(cnt) <= threshold_area]
            
                        # Calculate the tallest height for similarity comparison
                        height_list = [cv.boundingRect(cnt)[3] for cnt in digits_cnts]
                        
                        if len(height_list) == 0:
                            # print("No valid digit contours found. Skipping...")
                            continue
                        
                        tallest_height = max(height_list)
                        
                        # Set the threshold for similarity comparison
                        threshold = 0.35 # Maximum percentage difference allowed
            
                        # Create a dictionary to store the segment dictionaries for each contour
                        contour_dict = {}
            
                        # Loop over the sorted digits contours
                        for i, cnt in enumerate(sorted(digits_cnts, key=lambda c: cv.boundingRect(c)[0])):
                            (x, y, w, h) = cv.boundingRect(cnt)
                            roi = eroded[y: y + h, x: x + w]
                        
                            # Calculate the height threshold for similarity based on the tallest height
                            height_threshold = threshold * tallest_height
            
                            # Check if the current height is similar to the tallest height
                            if abs(tallest_height - h) <= height_threshold:
                                # Create a segment dictionary for each contour
                                segment_dict = {'segment': roi, 'width': w}
            
                                # Add the segment dictionary to the contour dictionary with contour number as the key
                                contour_dict[f'contour{i}'] = segment_dict
            
                        # Add this image's contour dictionary to all_segment_dicts
                        all_segment_dicts[filename] = contour_dict

                    #SAVE IMAGES OF NUMBERS 
            
                    # Set the desired size of the saved images
                    saved_image_size = (32, 32)
                    # Iterate over each segment image in all_segment_dicts
                    for contour_idx, (contour, segment_dict) in enumerate(all_segment_dicts.items()):
                        current_contour_dict = {}
                        # Check for an invalid number of segments, if yes, continue to next iteration
                        if segment_dict is None or len(segment_dict) <= 2 or len(segment_dict) >= 7:
                            continue
                        
                        # Calculate the average width for this particular contour's segments
                        widths = [segment['width'] for segment in segment_dict.values()]
                        # Calculate the 3/4 width
                        three_quarter_width = (3 * sum(widths)) // (4 * len(widths)) if len(widths) > 0 else 0
            
                        # Iterate over the segments in this contour
                        for img_idx, segment in enumerate(segment_dict.values()):
                            width = segment['width']
                            # Convert width to integer
                            width = int(float(width))
            
                            # Choose the canvas size based on the width of the segment relative to the average width
                            if width < three_quarter_width:
                                canvas_size = (20, 105)
                            else:
                                canvas_size = (20, 38)
            
                            # Resize the segment image to a smaller size
                            resized_segment = cv.resize(segment['segment'], (saved_image_size[1] // 2, saved_image_size[0] // 2))
            
                            # Apply denoising using Non-local Means Denoising
                            denoised_segment = cv.fastNlMeansDenoising(resized_segment, h=20)
            
                            # Apply image sharpening using the Unsharp Mask filter
                            blurred_segment = cv.GaussianBlur(denoised_segment, (5, 5), 0)
                            sharpened_segment = cv.addWeighted(denoised_segment, 2.5, blurred_segment, -1.5, 0)
            
                            # Create a white canvas with the desired canvas size
                            canvas = np.ones(canvas_size, dtype=np.uint8) * 255
            
                            # Calculate the position to paste the sharpened segment on the canvas
                            paste_x = (canvas_size[1] - sharpened_segment.shape[1]) // 2
                            paste_y = (canvas_size[0] - sharpened_segment.shape[0]) // 2
            
                            # Invert the colors of the sharpened segment (black to white, white to black)
                            inverted_segment = cv.bitwise_not(sharpened_segment)
            
                            # Paste the inverted segment image on the canvas
                            canvas[paste_y:paste_y+sharpened_segment.shape[0], paste_x:paste_x+sharpened_segment.shape[1]] = inverted_segment
            
                            # Resize the canvas to the final image size
                            resized_canvas = cv.resize(canvas, saved_image_size)
            
                            # Normalize the pixel values between 0 and 1
                            normalized_canvas = resized_canvas / 255.0
            
                            # print(f"Saved segment {img_idx+1} of contour {contour_idx+1} as {filename}")
                            if segment_dict is None or len(segment_dict) <= 2 or len(segment_dict) >= 7:
                                # print(f"Invalid segment for contour {contour}. Skipping...")
                                continue
                            images_saved_per_contour[contour] = len(segment_dict)
                            # Create the filename key for the dictionary
                            key_name = f"{contour_idx+1}_{img_idx+1}"
                            
                            # Instead of saving to a file, store in the dictionary
                            current_contour_dict[key_name] = resized_canvas
                        
                        # After processing all segments of a contour
                        all_contours_list.append(current_contour_dict)
            
                    return all(len(d) >= 2 for d in all_contours_list)
            
                # Main execution
                dir_path = "path_to_your_images"
                kernel_size = (4, 5)
                success = process_images(kernel_size)
                if not success:
                    # Retry with new kernel size if less than two images are saved per contour
                    new_kernel_size = (7, 9)
                    success = process_images(new_kernel_size)            
            
                # %%
                #1
                #LOAD IMAGES AND LET MODEL PREDICT
                import os
                model_path = "shvn_model.tflite"
                if not os.path.exists(model_path):
                    print(f"ERROR: Model file does not exist at {model_path}")
                else:
                    interpreter = tflite.Interpreter(model_path=model_path)
                    interpreter.allocate_tensors()

                # Load the TFLite model and allocate tensors
                try:
                    interpreter = tflite.Interpreter(model_path="shvn_model.tflite")
                    interpreter.allocate_tensors()
                except Exception as e:
                    print(f"ERROR: Failed to initialize the interpreter: {e}")
                # Get input and output details
                input_details = interpreter.get_input_details()
                output_details = interpreter.get_output_details()

                # Create dictionaries to store predictions and probabilities for each contour
                contour_predictions = {}
                contour_probabilities = {}
                def softmax(x):
                    e_x = np.exp(x - np.max(x))  # subtract max to stabilize
                    return e_x / e_x.sum(axis=0)
            
                # Iterate over each dictionary (which corresponds to a contour) in the all_contours_list
                for contour_dict in all_contours_list:
                    # Extract the contour index from one of the keys (since they all start with the contour index)
                    contour_idx = int(float(list(contour_dict.keys())[0].split("_")[0]))
            
                    # Iterate over each image stored in the current dictionary (contour)
                    for key, img in contour_dict.items():
            
                        # Preprocess the image
                        img = cv.resize(img, (32, 32))
                        img = np.expand_dims(img, axis=-1)
                        img = img.astype(np.float32) / 255.0
                        img = np.repeat(img, 3, axis=-1)
            
                        # Set the input tensor for the interpreter
                        interpreter.set_tensor(input_details[0]['index'], img[np.newaxis, ...])
            
                        # Invoke the interpreter
                        interpreter.invoke()
                        
                        # Get the output tensor from the interpreter
                        prediction = interpreter.get_tensor(output_details[0]['index'])
                        probabilities = probabilities = softmax(prediction)[0] #tf.nn.softmax(prediction).numpy()[0]
            
                        # Get the predicted label and its probability
                        predicted_label = np.argmax(prediction)
                        probability = probabilities[predicted_label]
            
                        # Append predictions and probabilities to the dictionaries
                        if contour_idx not in contour_predictions:
                            contour_predictions[contour_idx] = []
                        if probability >= 0.14:
                            contour_predictions[contour_idx].append(predicted_label)
                        if contour_idx not in contour_probabilities:
                            contour_probabilities[contour_idx] = []
                        contour_probabilities[contour_idx].append(probability)
                        
                # Remap keys of the contour_predictions_2 dictionary to start from 0
                keys = sorted(contour_predictions.keys())
                contour_predictions = {idx: contour_predictions[key] for idx, key in enumerate(keys)}
            
                # %%
                #1
                #PRINT PRICES
            
                # Filter out keys that have more than 5 values
                invalid_keys = [key for key, values in contour_predictions.items() if len(values) > 5]
            
            
                # Ensure both lists have the same number of elements
                num_predictions = len(contour_predictions)
                num_definitions = len(sorted_definition_list)
            
                if num_predictions < num_definitions:
                    # Add empty strings to contour_predictions to match the length of sorted_definition_list
                    contour_predictions.update({i: '' for i in range(num_predictions, num_definitions)})
            
                results = {}
            
                # Iterate over each match in sorted_definition_list
                for idx, definition in enumerate(sorted_definition_list):
                    if idx in invalid_keys:
                        results[definition] = "Price not found"
                        continue
            
                    predictions_str = contour_predictions[idx]  # Get the string representation
            
                    # Convert the string to a list of integers
                    predictions = [int(float(char)) for char in predictions_str]
            
                    # Find the index of the first occurrence of 1, 2, or 4 in the list
                    index_1 = predictions.index(1) if 1 in predictions else -1
                    index_2 = predictions.index(2) if 2 in predictions else -1
                    index_4 = predictions.index(4) if 4 in predictions else -1
            
                    # Create a list of indexes and remove -1
                    indexes = [index for index in [index_1, index_2, index_4] if index != -1]
            
                    if indexes:
                        # Choose the smallest index
                        index = min(indexes)
            
                        # Check if at least three numbers are available
                        if len(predictions) - index < 3:
                            results[definition] = "Price not found"
                            continue
            
                        # Extract three consecutive digits from the 'predictions' list, starting from the index obtained from 'indexes'
                        digits = predictions[index: index+3]
            
                        digits_str = ''.join(map(str, digits))
            
                        # Convert the formatted string to a float and divide by 100 to get two decimal places
                        number = "{:.2f}".format(float(digits_str) / 100)
            
                        if 1.10 <= float(number) <= 2.90:
                            results[definition] = number
                        else:
                            results[definition] = "Price not found" 
                    else:
                        results[definition] = "Price not found"
            
                # print(results)
            
                # Remove "Price not found" entries from the results dictionary
                results_1 = {definition: price for definition, price in results.items() if price != "Price not found"}
                # print(results_1)
            
                # Check if the first key is '1' and assign 'Diesel' to the highest value and 'Benzin' to the rest
                if results_1:
                    if list(results_1.keys())[0] == 1:
                        max_value = max(results_1.values())
                        results_1 = {'Diesel': max_value}
                        for key, value in results_1.items():
                            if value != max_value:
                                results_1['Benzin'] = value
                                break
                    else:
                        results_1 = results_1
            
                    # print(results_1)
                else:
                    results_1 = {}
                    #print("No results found.")
            
                # %%
                #2
                #SELECT CONTOURS BY LENGHT OF TEXT LIST
            
                # Create a dictionary to store the images
                image_dict_pre_2 = {}
            
                # Get the current height of the image
                height = img_original.shape[0]
            
                # Define the minimum and maximum heights
                min_height = 500
                max_height = 800
            
                # Calculate the scaling factor based on the desired height range
                scale_factor = min(max(min_height / height, 1), max_height / height)
            
                # Resize the image using the calculated scale factor
                img_color = cv.resize(img_original, None, None, fx=scale_factor, fy=scale_factor)
            
                # Function to split the image vertically into equal parts
                def split_equally(image, num_parts):
                    height = image.shape[0]
                    part_height = height // num_parts
            
                    splits = []
                    for i in range(num_parts):
                        start = i * part_height
                        end = (i + 1) * part_height
                        splits.append(image[start:end, :])
            
                    return splits
            
                # Function to enhance the resolution of an image and crop 2/7 from the left
                def enhance_resolution(image_array, scale_factor, crop_fraction=2/7):
                    image = Image.fromarray(image_array)  # Convert the image array to a PIL Image
            
                    # Calculate the new width and height based on the scale factor
                    width = image.width * scale_factor
                    height = image.height * scale_factor
            
                    # Calculate the number of pixels to crop from the left
                    crop_pixels = int(float(image.width * crop_fraction))
            
                    # Crop the image from the left
                    cropped_image = image.crop((crop_pixels, 0, image.width, image.height))
            
                    # Resize the image using the BICUBIC interpolation method
                    resized_image = cropped_image.resize((int(float(width)), int(float(height))), Image.BICUBIC)
            
                    enhanced_image_array = np.array(resized_image)  # Convert the resized image back to a numpy array
            
                    return enhanced_image_array  # Return the enhanced image array
            
                # Specify the scale factor for image enhancement
                scale_factor = 2  # Increase resolution by a factor of 2
            
                # Function to convert image to black and white
                def convert_to_black_and_white(image):
                    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
                    # _, binary_image = cv.threshold(gray, 120, 255, cv.THRESH_BINARY)
                    return gray
            
                if not sorted_definition_list:
                    pass
                    # You can exit or continue with other operations
                else:
                    # Split the image into parts equal to the length of sorted_definition_list
                    split_images = split_equally(img_color, num_parts=len(sorted_definition_list))
            
                    # Convert each split image to black and white and save in the "contour_images_2" folder
                    output_folder = "contour_images_2"
                    os.makedirs(output_folder, exist_ok=True)
            
                    for i, split_image in enumerate(split_images):
                        enhanced_image = enhance_resolution(split_image, scale_factor)  # Enhance the resolution and crop  from the left
                        black_and_white_image = convert_to_black_and_white(enhanced_image)   # Convert to black and white
                        cv.imwrite(os.path.join(output_folder, f"split_part_{i+1}.png"), black_and_white_image)
                        # Wait for a key press and then close all windows
                        # cv.imshow("contours_2", black_and_white_image)
                        # cv.waitKey(0)
            
                # %%
                #2
                #CONTAIN EVERY OBJECT IN A RECTANGLE
                #IF NUMBERS SAVED ARE LESS THAN 2 PER CONTOUR TRY WITH DIFFERENT MORPHING SETTINGS
            
                all_contours_list = []
            
                def process_images(kernel_size):
                    all_segment_dicts = {} 
                    images_saved_per_contour = {}  
            
                    # Loop over the keys and values in the image_dict_pre_2
                    for filename, roi in image_dict_pre_2.items():  # Note: changed roi_color to roi
                        
                        # cv.imshow("contours_2", roi)
                        # cv.waitKey(0)
            
                        # Define the minimum and maximum heights
                        min_height = 800
                        max_height = 1200
            
                        # Calculate the scaling factor based on the desired height range
                        scale_factor = min(max(min_height / height, 1), max_height / height)
            
                        # Resize the image using the calculated scale factor
                        roi = cv.resize(roi, None, None, fx=scale_factor, fy=scale_factor)
                        
                        # Convert the image to grayscale if needed
                        if len(roi.shape) > 2:
                            gray_roi = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
                        else:
                            gray_roi = roi
            
                        # Apply adaptive thresholding
                        _, edged = cv.threshold(
                            gray_roi, 200, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU
                        )
            
                        edged = 255 - edged
            
                        # Apply the kernel
                        kernel = cv.getStructuringElement(cv.MORPH_RECT, kernel_size)
                        dilated = cv.dilate(edged, kernel, iterations=2)
            
                        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (1, 3),)
                        eroded = cv.erode(dilated, kernel, iterations=1)
                        # cv.imshow("Eroded", eroded)
                        # cv.waitKey(0)
            
                        h = roi.shape[0]
                        ratio = int(float(h * 0.07))
                        eroded[-ratio:,] = 0
                        eroded[:, :ratio] = 0
            
                        cnts, _ = cv.findContours(eroded, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
                        
                        canvas = eroded.copy()
                        digits_cnts = []
                        
                        for cnt in cnts:
                            (x, y, w, h) = cv.boundingRect(cnt)
                            if h > 20:
                                digits_cnts += [cnt]
                                cv.drawContours(canvas, [cnt], -1, (75, 0, 130), thickness=2)
                                cv.rectangle(canvas, (x, y), (x + w, y + h), (75, 0, 130), thickness=2)
            
                        # print(f"No. of Digit Contours: {len(digits_cnts)}")
            
                        # Calculate the height and width of the image
                        img_height, img_width = roi.shape[:2]
                        
                        # Thresholds for the contours
                        height_threshold = 0.85 * img_height
                        width_threshold = 0.85 * img_width
            
                        # Calculate the threshold area for the largest contour to be removed
                        threshold_area = 0.1 * img_height * img_width
            
                        # Remove contours that are too large
                        digits_cnts = [cnt for cnt in digits_cnts 
                                    if cv.boundingRect(cnt)[3] <= height_threshold 
                                    and cv.boundingRect(cnt)[2] <= width_threshold 
                                    and cv.contourArea(cnt) <= threshold_area]
            
                        # Print the number of digit contours after removing contours larger than the threshold
                        # print(f"No. of Digit Contours after removing large contours: {len(digits_cnts)}")
            
                        # Calculate the tallest height for similarity comparison
                        height_list = [cv.boundingRect(cnt)[3] for cnt in digits_cnts]
                        
                        if len(height_list) == 0:
                            # print("No valid digit contours found. Skipping...")
                            continue
                        
                        tallest_height = max(height_list)
                        
                        # Set the threshold for similarity comparison
                        threshold = 0.35 # Maximum percentage difference allowed
            
                        # Create a dictionary to store the segment dictionaries for each contour
                        contour_dict = {}
            
                        # Loop over the sorted digits contours
                        for i, cnt in enumerate(sorted(digits_cnts, key=lambda c: cv.boundingRect(c)[0])):
                            (x, y, w, h) = cv.boundingRect(cnt)
                            roi = eroded[y: y + h, x: x + w]
                        
                            # Calculate the height threshold for similarity based on the tallest height
                            height_threshold = threshold * tallest_height
            
                            # Check if the current height is similar to the tallest height
                            if abs(tallest_height - h) <= height_threshold:
                                # Create a segment dictionary for each contour
                                segment_dict = {'segment': roi, 'width': w}
            
                                # Add the segment dictionary to the contour dictionary with contour number as the key
                                contour_dict[f'contour{i}'] = segment_dict
            
                        # Add this image's contour dictionary to all_segment_dicts
                        all_segment_dicts[filename] = contour_dict

                    
                    #SAVE IMAGES OF NUMBERS 
            
                    # Set the desired size of the saved images
                    saved_image_size = (32, 32)
            
                    # Iterate over each segment image in all_segment_dicts
                    for contour_idx, (contour, segment_dict) in enumerate(all_segment_dicts.items()):
                        current_contour_dict = {}
            
                        # Check for an invalid number of segments, if yes, continue to next iteration
                        if segment_dict is None or len(segment_dict) <= 2 or len(segment_dict) >= 7:
                            continue
            
                        # Calculate the average width for this particular contour's segments
                        widths = [segment['width'] for segment in segment_dict.values()]
                        three_quarter_width = (3 * sum(widths)) // (4 * len(widths)) if len(widths) > 0 else 0
            
                        # Iterate over the segments in this contour
                        for img_idx, segment in enumerate(segment_dict.values()):
                            width = segment['width']
                            # Convert width to integer
                            width = int(float(width))
            
                            # Choose the canvas size based on the width of the segment relative to the average width
                            if width < three_quarter_width:
                                canvas_size = (20, 105)
                            else:
                                canvas_size = (20, 38)
            
                            # Resize the segment image to a smaller size
                            resized_segment = cv.resize(segment['segment'], (saved_image_size[1] // 2, saved_image_size[0] // 2))
            
                            # Apply denoising using Non-local Means Denoising
                            denoised_segment = cv.fastNlMeansDenoising(resized_segment, h=20)
            
                            # Apply image sharpening using the Unsharp Mask filter
                            blurred_segment = cv.GaussianBlur(denoised_segment, (5, 5), 0)
                            sharpened_segment = cv.addWeighted(denoised_segment, 2.5, blurred_segment, -1.5, 0)
            
                            # Create a white canvas with the desired canvas size
                            canvas = np.ones(canvas_size, dtype=np.uint8) * 255
            
                            # Calculate the position to paste the sharpened segment on the canvas
                            paste_x = (canvas_size[1] - sharpened_segment.shape[1]) // 2
                            paste_y = (canvas_size[0] - sharpened_segment.shape[0]) // 2
            
                            # Invert the colors of the sharpened segment (black to white, white to black)
                            inverted_segment = cv.bitwise_not(sharpened_segment)
            
                            # Paste the inverted segment image on the canvas
                            canvas[paste_y:paste_y+sharpened_segment.shape[0], paste_x:paste_x+sharpened_segment.shape[1]] = inverted_segment
            
                            # Resize the canvas to the final image size
                            resized_canvas = cv.resize(canvas, saved_image_size)
            
                            # Normalize the pixel values between 0 and 1
                            normalized_canvas = resized_canvas / 255.0
            
                            # print(f"Saved segment {img_idx+1} of contour {contour_idx+1} as {filename}")
                            if segment_dict is None or len(segment_dict) <= 2 or len(segment_dict) >= 7:
                                # print(f"Invalid segment for contour {contour}. Skipping...")
                                continue
                            images_saved_per_contour[contour] = len(segment_dict)
                            # Create the filename key for the dictionary
                            key_name = f"{contour_idx+1}_{img_idx+1}"
                            
                            # Instead of saving to a file, store in the dictionary
                            current_contour_dict[key_name] = resized_canvas
                        
                        # After processing all segments of a contour
                        all_contours_list.append(current_contour_dict)
            
                    return all(len(d) >= 2 for d in all_contours_list)
            
                # Main execution
                dir_path = "path_to_your_images"
                kernel_size = (10, 10)
            
                success = process_images(kernel_size)
                if not success:
                    # Retry with new kernel size if less than two images are saved per contour
                    new_kernel_size = (3, 4)
                    success = process_images(new_kernel_size)    
            
                # %%
                #2
                #LOAD IMAGES AND LET MODEL PREDICT
            
                # Load the TFLite model and allocate tensors
                # interpreter = tflite.Interpreter(model_path=os.path.join(App.get_running_app().user_data_dir, 'app', 'shvn_model.tflite'))
                interpreter = tflite.Interpreter(model_path="shvn_model.tflite")
                interpreter.allocate_tensors()
            
                # Get input and output details
                input_details = interpreter.get_input_details()
                output_details = interpreter.get_output_details()
            
                # Create dictionaries to store predictions and probabilities for each contour
                contour_predictions_2 = {}
                contour_probabilities_2 = {}
            
                # Iterate over each dictionary (which corresponds to a contour) in the all_contours_list
                for contour_dict in all_contours_list:
            
                    # Extract the contour index from one of the keys (since they all start with the contour index)
                    contour_idx = int(float(list(contour_dict.keys())[0].split("_")[0]))
            
                    # Iterate over each image stored in the current dictionary (contour)
                    for key, img in contour_dict.items():
            
                        # Preprocess the image
                        img = cv.resize(img, (32, 32))
                        img = np.expand_dims(img, axis=-1)
                        img = img.astype(np.float32) / 255.0
                        img = np.repeat(img, 3, axis=-1)
            
                        # Set the input tensor for the interpreter
                        interpreter.set_tensor(input_details[0]['index'], img[np.newaxis, ...])
            
                        # Invoke the interpreter
                        interpreter.invoke()
            
                        # Get the output tensor from the interpreter
                        prediction = interpreter.get_tensor(output_details[0]['index'])
                        probabilities = probabilities = softmax(prediction)[0] #tf.nn.softmax(prediction).numpy()[0]
            
                        # Get the predicted label and its probability
                        predicted_label = np.argmax(prediction)
                        probability = probabilities[predicted_label]
            
                        # Append predictions and probabilities to the dictionaries
                        if contour_idx not in contour_predictions_2:
                            contour_predictions_2[contour_idx] = []
                        if probability >= 0.14:
                            contour_predictions_2[contour_idx].append(predicted_label)
                        if contour_idx not in contour_probabilities_2:
                            contour_probabilities_2[contour_idx] = []
                        contour_probabilities_2[contour_idx].append(probability)
                        
                # Remap keys of the contour_predictions_2 dictionary to start from 0
                keys = sorted(contour_predictions_2.keys())
                contour_predictions_2 = {idx: contour_predictions_2[key] for idx, key in enumerate(keys)}
            
                # %%
                #2
                #PRINT PRICES
                # Filter out keys that have more than 5 values
                invalid_keys = [key for key, values in contour_predictions_2.items() if len(values) > 5]
            
                # Ensure both lists have the same number of elements
                num_predictions = len(contour_predictions_2)
                num_definitions = len(sorted_definition_list)
            
                if num_predictions < num_definitions:
                    # Add empty strings to contour_predictions_2 to match the length of sorted_definition_list
                    contour_predictions_2.update({i: '' for i in range(num_predictions, num_definitions)})
            
                resultss = {}
            
            
                # Iterate over each match in sorted_definition_list
                for idx, definition in enumerate(sorted_definition_list):
                    if idx in invalid_keys:
                        results[definition] = "Price not found"
                        continue
            
                    predictions_str = contour_predictions_2[idx]  # Get the string representation
            
                    # Convert the string to a list of integers
                    predictions = [int(float(char)) for char in predictions_str]
            
                    # Find the index of the first occurrence of 1, 2, or 4 in the list
                    index_1 = predictions.index(1) if 1 in predictions else -1
                    index_2 = predictions.index(2) if 2 in predictions else -1
                    index_4 = predictions.index(4) if 4 in predictions else -1
            
                    # Create a list of indexes and remove -1
                    indexes = [index for index in [index_1, index_2, index_4] if index != -1]
            
                    if indexes:
                        # Choose the smallest index
                        index = min(indexes)
            
                        # Convert the digits starting from the first '1', '2' or '4' to a string
                        digits = predictions[index: index+3]
            
                        # If the first digit is a '4', change it to a '1'
                        if digits[0] == 4:
                            digits[0] = 1
            
                        digits_str = ''.join(map(str, digits))
            
                        # Convert the formatted string to a float and divide by 100 to get two decimal places
                        number = "{:.2f}".format(float(digits_str) / 100)
            
                        if '1.10' <= number <= '2.90':
                            resultss[definition] = number
                        else:
                            resultss[definition] = "Price not found"  # Price not in range (1.50 - 2.30)
                # print(resultss)
                # Remove "Price not found" entries from the resultss dictionary
                resultss_1 = {definition: price for definition, price in resultss.items() if price != "Price not found"}
                # print(resultss_1)
            
                # Check if the first key is '1' and assign 'Diesel' to the highest value and 'Benzin' to the rest
                if resultss_1:
                    if list(resultss_1.keys())[0] == 1:
                        max_value = max(resultss_1.values())
                        results_2 = {'Diesel': max_value}
                        for key, value in resultss_1.items():
                            if value != max_value:
                                results_2['Benzin'] = value
                                break
                    else:
                        results_2 = resultss_1
            
                    # print(results_2)
                else:
                    results_2 = {}
                    #print("No results found.")
                # print(results_2)
            
                # %%
                #3
                # Calculate the average y-coordinate of all boxes
                avg_ys = []  
                for box in boxes:
                    ys = [coord[1] for coord in box]
                    avg_ys.append(np.mean(ys))
            
                # Pair each box with its corresponding word and average y-coordinate
                pairs = list(zip(boxes, words, avg_ys))
            
                # Sort the pairs based on the average y-coordinate
                pairs.sort(key=lambda pair: pair[2])
            
                # Initialize a list to hold the final joined words
                final_words = []
            
                # Process each pair
                for i in range(len(pairs)):
                    current_word = pairs[i][1]
                    current_avg_y = pairs[i][2]
                    
                    # Check if this is the first pair, add its word to final_words
                    if i == 0:
                        final_words.append(current_word)
                    else:
                        # Check if the current word contains a numeric value (no spaces) and no letters
                        has_numeric_value = any(char.isdigit() for char in current_word)
                        has_letters = any(char.isalpha() for char in current_word)
                        
                        if has_numeric_value and not has_letters:
                            # Check the last word added to final_words to prevent consecutive numbers
                            last_word = final_words[-1]
                            if not any(char.isdigit() for char in last_word):
                                final_words[-1] += ' ' + current_word
                        else:
                            # Check the last word added to final_words to prevent consecutive words
                            last_word = final_words[-1]
                            if any(char.isdigit() for char in last_word):
                                final_words.append(current_word)
                            else:
                                # Remove the last word if it's a duplicate and add the current word
                                final_words[-1] = current_word

                # Remove duplicates if they occur due to the above process
                for i in range(len(final_words)-2, 0, -1):
                    if final_words[i-1] == final_words[i+1]:
                        final_words[i-1] = final_words[i-1] + ' ' + final_words[i]
                        del final_words[i]

                sorted_definition_list = final_words
                # Check if sorted_definition_list is empty
                if not sorted_definition_list:
                    result_list = []
                else:
                    result_list = [sorted_definition_list[0]]
                    for i in range(1, len(sorted_definition_list)):
                        current_word = sorted_definition_list[i]
                        previous_word = sorted_definition_list[i - 1]

                        # Check if the current word is the same as the previous word
                        if current_word == previous_word and i > 1:
                            # If the same, append the current word to the previous word in result_list
                            result_list[-2] += ' ' + current_word
                        else:
                            result_list.append(current_word)

                    sorted_definition_list = result_list

            
                # print(sorted_definition_list)
                # Lists to store words based on criteria
                List1 = []  # For words without 3 or more numbers
                List2 = []  # For words with 3 or more numbers
            
                for word in words:
                    # Extract all numbers (including decimals)
                    numbers = re.findall(r'\d+\.\d+|\d+', word)
                    
                    # Count total digits in all numbers found in the word
                    total_digits = sum(len(num.replace('.', '')) for num in numbers)
            
                    if total_digits >= 3:
                        List2.append(word)
                    else:
                        List1.append(word)
                        
                if not List1 or not List2:  # Check if either List1 or List2 is empty
                    results_3 = {}
                else:
            
                    # Pair words and numbers with their bounding boxes and y-coordinates
                    words_with_coords = list(zip(words, boxes, avg_ys))
            
                    # Separate words and numbers based on List1 and List2
                    words_coords = [item for item in words_with_coords if item[0] in List1]
                    numbers_coords = [item for item in words_with_coords if item[0] in List2]
            
                    results_3 = {}
            
                    # If img_original is a numpy ndarray
                    img_height, img_width = img_original.shape[:2]
            
                    # Set the threshold as a percentage of the image height
                    percentage = 5  # Change this value as needed
                    threshold = (percentage / 100) * img_height
            
                    for word, word_box, word_y in words_coords:
                        # Find the closest number to the current word based on y-coordinate
                        closest_number = min(numbers_coords, key=lambda num: abs(word_y - num[2]))
                        
                        # Only pair the word with the number if their y-coordinate difference is within the threshold
                        if abs(word_y - closest_number[2]) < threshold:
                            if closest_number[0] not in results_3.values():  # Check if the number is already matched
                                results_3[word] = closest_number[0]
            
                    # print(results_3)
            
                    # Update numbers in the dictionary based on the specified condition
                    # Adjust the numbers in the dictionary based on the specified condition
                    for key in results_3:
                        value = results_3[key]
                        
                        # If the number starts with a dot and the first number after the dot is 4 or greater
                        if value.startswith('.') and int(float(value[1])) >= 4:
                            value = '1' + value
                        elif value.startswith('.'):
                            value = '0' + value
                        
                        # If the number doesn't contain a dot
                        if value[0].isdigit() and '.' not in value:
                            if int(float(value[0])) in [1, 2, 3]:  # First digit is 1, 2, or 3
                                value = value[0] + '.' + value[1:]
                            elif int(float(value[0])) >= 4:  # First digit is 4 or greater
                                value = '1.' + value
                        
                        # Convert the number to float and round to two decimal places
                        converted_num = round(float(value), 2)
                        results_3[key] = "{:.2f}".format(converted_num)
                        
                #4
                # Calculate the average y-coordinate of all boxes
                avg_ys = []
                for box in boxes:
                    ys = [coord[1] for coord in box]
                    avg_ys.append(np.mean(ys))

                # Pair each box with its corresponding word and average y-coordinate
                pairs = list(zip(boxes, words, avg_ys))

                # Sort the pairs based on the average y-coordinate
                pairs.sort(key=lambda pair: pair[2])

                # Initialize a list to hold the final joined words
                final_words = []

                # Process each pair
                for i in range(len(pairs)):
                    current_word = pairs[i][1]
                    current_avg_y = pairs[i][2]

                    # Check if this is the first pair, add its word to final_words
                    if i == 0:
                        final_words.append(current_word)
                    else:
                        # Check if the current word contains a numeric value (no spaces) and no letters
                        has_numeric_value = any(char.isdigit() for char in current_word)
                        has_letters = any(char.isalpha() for char in current_word)

                        if has_numeric_value and not has_letters:
                            # Check the last word added to final_words to prevent consecutive numbers
                            last_word = final_words[-1]
                            if not any(char.isdigit() for char in last_word):
                                final_words[-1] += ' ' + current_word
                        else:
                            # Check the last word added to final_words to prevent consecutive words
                            last_word = final_words[-1]
                            if any(char.isdigit() for char in last_word):
                                final_words.append(current_word)
                            else:
                                # Replace the last word if it's a duplicate and add the current word
                                final_words[-1] = current_word

                # Remove duplicates if they occur due to the above process
                for i in range(len(final_words) - 2, 0, -1):
                    if final_words[i - 1] == final_words[i + 1]:
                        final_words[i - 1] = final_words[i - 1] + ' ' + final_words[i]
                        del final_words[i]

                sorted_definition_list = final_words
                # Check if sorted_definition_list is empty
                if not sorted_definition_list:
                    result_list = []
                else:
                    result_list = [sorted_definition_list[0]]
                    for i in range(1, len(sorted_definition_list)):
                        current_word = sorted_definition_list[i]
                        previous_word = sorted_definition_list[i - 1]

                        # Check if the current word is the same as the previous word
                        if current_word == previous_word and i > 1:
                            # If the same, append the current word to the previous word in result_list
                            result_list[-2] += ' ' + current_word
                        else:
                            result_list.append(current_word)

                    sorted_definition_list = result_list


                # Debugging output
                # print("Final sorted definition list:", sorted_definition_list)

                # Lists to store words based on criteria
                List1 = []  # For words without 3 or more numbers
                List2 = []  # For words with 3 or more numbers

                for word in words:
                    # Extract all numbers (including decimals)
                    numbers = re.findall(r'\d+\.\d+|\d+', word)

                    # Count total digits in all numbers found in the word
                    total_digits = sum(len(num.replace('.', '')) for num in numbers)

                    if total_digits >= 3:
                        List2.append(word)
                    else:
                        List1.append(word)

                # Check if either List1 or List2 is empty
                if not List1 or not List2:
                    results_4 = {}
                else:
                    # Create the dictionary by pairing elements from List1 and List2
                    results_4 = dict(zip(List1, List2))
                    
                    # Ensure each value in results_4 has exactly two decimal places
                    for key in results_4:
                        try:
                            value = float(results_4[key])
                            results_4[key] = "{:.2f}".format(value)
                        except ValueError:
                            # Handle cases where the value cannot be converted to float
                            results_4[key] = "0.00" 


                # %%
                def find_longest_dictionary(a, b, c, d):
                    # Check if numbers in the dictionary meet the required conditions
                    def is_valid(d):
                        unique_numbers = set()
                        for value in d.values():
                            try:
                                number = float(value)
                                decimal_part = str(value).split('.')
                                if not (1 <= number <= 500) or (len(decimal_part) == 2 and len(decimal_part[1]) != 2):
                                    return False
                                unique_numbers.add(number)
                            except ValueError:
                                return False

                        # Checking if at least two numbers in the dictionary are different
                        if len(unique_numbers) < 2:
                            return False

                        return True

                    valid_dicts = {
                        'a': a,
                        'b': b,
                        'c': c,
                        'd': d
                    }
                    valid_dicts = {key: value for key, value in valid_dicts.items() if is_valid(value)}
                    
                    if not valid_dicts:
                        return None, None

                    winner_key = max(valid_dicts, key=lambda k: len(valid_dicts[k]))
                    return valid_dicts[winner_key], winner_key

                # Finding the longest valid dictionary and its identifier
                longest_dict, winner = find_longest_dictionary(results_3, results_1, results_2, results_4)
                if longest_dict:
                    print(longest_dict)
                    return longest_dict  
                else:
                    return {"error": "invalid"}
            else:
                return {"error": "no path"}

    except Exception as e:
        print(f"DEBUG ERROR: Exception occurred - {e}")
        # raise e
        # Instead of raising the exception, return an error message in a dictionary
        return {"error": "bad quality"}

