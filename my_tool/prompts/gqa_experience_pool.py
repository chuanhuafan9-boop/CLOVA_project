FAILED_SUBQUESTION=[
"""Question: Are there any windows in the picture that are not rectangular?
SubQuesion:
Step1, Locate windows in the given image, and obtain bounding boxes of windows.
Step2, Crop the image region of windows from the given image, based on bounding boxes of windows. The bounding boxes are obtained in Step1.
Step3, Asking the image region of windows, 'What shape is the window?'. The image region of windows is obtained in Step2.
Step4, Determine whether the answer is 'yes' or 'no' by executing Python expression, based on the intermediate answers obtained in Step3. If the shape is not equal to 'rectangular', the answer is 'yes'; On the contrary, the answer is 'no'.
Step5, Visualize results.
Reason:
In Step3 of the subquestions, the subquestions should ask 'Are the windows rectangular?', instead of asking 'What shape is the window?'. Then, the subquestions should further determine whether the answer is 'yes' or 'no' by executing Python expression, based on the intermediate answers obtained in Step3. If the answer is 'yes', the answer is 'no'; On the contrary, the answer is 'yes'.
""",
]


FAILED_PROGRAM=[
"""
Question: Is the lamp different in color than the shirt?
SubQuesion:
Step1, Locate the lamp in the given image, and obtain bounding boxes of lamp.
Step2, Crop the region of the lamp from the given image, based on bounding boxes of lamp. The bounding boxes are obtained in Step1.
Step3, Asking the image region of lamp, 'What color is the lamp?'. The image region of lamp is cropped in Step2.
Step4, Locate the shirt in the given image, and obtain bounding boxes of shirt. 
Step5, Crop the region of the shirt from the given image, based on bounding boxes of shirt. The bounding boxes are obtained in Step4.
Step6, Asking the image region of shirt, 'What color is the shirt?'. The image region of lamp is cropped in Step5.
Step7, Determine whether the answer is 'yes' or 'no' by executing Python expression, based on the color of lamp and the color of shirt. The color of lamp and shirt is obtained in Step3 and Step6, respectively. If their color are the same, the answer is 'yes'; On the contrary, the answer is 'no'.
Step8, Visualize results.
Program:
BOX0=LOC(image=IMAGE,object='lamp')
IMAGE0=CROP(image=IMAGE,box=BOX0)
BOX1=LOC(image=IMAGE0,object='shirt')
ANSWER0=COUNT(box=BOX1)
ANSWER1=EVAL(expr="'yes' if {ANSWER0} > 0 else 'no'")
FINAL_RESULT=RESULT(var=ANSWER1)
Reason: The subquestions are correct, and can address the given question. But the Step3-Step6 of the program does not match the subquestion. The subquestions locate, crop, and ask color of the lamp and shirt, but the program counts the number of lamp.
""",
"""
Question: Is the lamp different in color than the shirt?
SubQuesion:
Step1, Locate the lamp in the given image, and obtain bounding boxes of lamp.
Step2, Crop the region of the lamp from the given image, based on bounding boxes of lamp. The bounding boxes are obtained in Step1.
Step3, Asking the image region of lamp, 'What color is the lamp?'. The image region of lamp is cropped in Step2.
Step4, Locate the shirt in the given image, and obtain bounding boxes of shirt. 
Step5, Crop the region of the shirt from the given image, based on bounding boxes of shirt. The bounding boxes are obtained in Step4.
Step6, Asking the image region of shirt, 'What color is the shirt?'. The image region of lamp is cropped in Step5.
Step7, Determine whether the answer is 'yes' or 'no' by executing Python expression, based on the color of lamp and the color of shirt. The color of lamp and shirt is obtained in Step3 and Step6, respectively. If their color are the same, the answer is 'yes'; On the contrary, the answer is 'no'.
Step8, Visualize results.
Program:
BOX0=LOC(image=IMAGE,object='lamp')
IMAGE0=CROP(image=IMAGE,box=BOX0)
BOX1=LOC(image=IMAGE0,object='shirt')
ANSWER0=COUNT(box=BOX1)
ANSWER1=EVAL(expr="'yes' if {ANSWER0} > 0 else 'no'")
FINAL_RESULT=RESULT(var=ANSWER1)
Reason: The subquestions are correct, and can address the given question. But the Step3-Step6 of the program does not match the subquestion. The subquestions locate, crop, and ask color of the lamp and shirt, but the program counts the number of lamp.
""",
"""
Question: Is the lamp different in color than the shirt?
SubQuesion:
Step1, Locate the lamp in the given image, and obtain bounding boxes of lamp.
Step2, Crop the region of the lamp from the given image, based on bounding boxes of lamp. The bounding boxes are obtained in Step1.
Step3, Asking the image region of lamp, 'What color is the lamp?'. The image region of lamp is cropped in Step2.
Step4, Locate the shirt in the given image, and obtain bounding boxes of shirt. 
Step5, Crop the region of the shirt from the given image, based on bounding boxes of shirt. The bounding boxes are obtained in Step4.
Step6, Asking the image region of shirt, 'What color is the shirt?'. The image region of lamp is cropped in Step5.
Step7, Determine whether the answer is 'yes' or 'no' by executing Python expression, based on the color of lamp and the color of shirt. The color of lamp and shirt is obtained in Step3 and Step6, respectively. If their color are the same, the answer is 'yes'; On the contrary, the answer is 'no'.
Step8, Visualize results.
Program:
BOX0=LOC(image=IMAGE,object='lamp')
IMAGE0=CROP(image=IMAGE,box=BOX0)
BOX1=LOC(image=IMAGE0,object='shirt')
ANSWER0=COUNT(box=BOX1)
ANSWER1=EVAL(expr="'yes' if {ANSWER0} > 0 else 'no'")
FINAL_RESULT=RESULT(var=ANSWER1)
Reason: The subquestions are correct, and can address the given question. But the Step3-Step6 of the program does not match the subquestion. The subquestions locate, crop, and ask color of the lamp and shirt, but the program counts the number of lamp.
""",
]


CURATED_SUBQUESTION=[
"""Question: Is the vehicle in the top of the image?
SubQuesion:
Step1, Locate the upper region of the given image since the question asks the top of the image, and obtain bounding boxes of the upper region.
Step2, Crop the upper region from the given image, based on bounding boxes of the upper region. The bounding boxes are obtained in Step1.
Step3, Locate vehicle in the upper region of the given image, and obtain bounding boxes of vehicle. The upper region is cropped in Step2.
Step4, Count the number of vehicle, based on bounding boxes of vehicle. The bounding boxes are obtained in Step3.
Step5, Determine whether the answer is 'yes' or 'no' by executing Python expression, based on the number of vehicles. The number is obtained in Step4. If the number is greater than zero, the answer is 'yes'; On the contrary, the answer is 'no'.
Step6, Visualize results.
""",
"""Question: What type of candy is in the bowl that the pizza cutter is to the right of?
Subquestion: 
Step1, Locate the pizza cutter, and obtain bounding boxes of the pizza cutter.
Step2, Crop the left part of the pizza cutter since the question is asking what is to the right of the pizza cutter. The bounding boxes are obtained in Step1.
Step3, Since the question is asking what type, then ask the image region, 'What candy is it?' image. The image is cropped in Step2.
Step4, Visualize results.
""",
"""Question: Are the glass bowls to the left of a book?
Subquestion: 
Step1, Locate the book, and obtain bounding boxes of the book.
Step2, Crop the left part of the book since the question is asking what is to the left of the book. The bounding boxes are obtained in Step1.
Step3, Try locate glass bowls in the cropped image. The image is cropped in Step2.
Step4, Count the number of bounding boxes. The bounding box is from Step3.
Step5, This is a yes or no question, so determine whether the answer is 'yes' or 'no' by executing Python expression.
Step6, Visualize results.
""",
"""Question: Does the cup that is to the right of the skateboarder look red?
Subquestion: 
Step1, Locate skateboarder in the given image, and obtain bounding boxes of skateboarder.
Step2, Crop the region right to the skateboarder, based on bounding boxes of skateboarder. The bounding boxes are obtained in Step1.
Step3, Asking the image region, 'What color is the cup?'. The image region of cup is obtained in Step2.
Step4, Determine whether the answer is 'yes' or 'no' by executing Python expression, based on the intermediate answers obtained in Step3. If the color is equal to 'red', the answer is 'yes'; On the contrary, the answer is 'no'.
Step5, Visualize results.
""",
"""Question: What's on the table?
Subquestion: 
Step1, Locate the table, and obtain bounding boxes of the table.
Step2, Crop the table since the question is asking what is on the table. The bounding boxes are obtained in Step1.
Step3, Ask the image region "what's on the table?". The image is cropped in Step2.
Step4, Visualize results.
""",
"""Question: Is the street light standing behind a truck?
SubQuesion:
Step1, Locate truck in the given image, and obtain bounding boxes of truck.
Step2, Crop the image region behind the truck from the given image, based on bounding boxes of truck. The bounding boxes are obtained in Step1.
Step3, Locate street light in the region behind the truck, and obtain bounding boxes of street light. The region behind the truck is cropped in Step2.
Step4, Count the number of street light, based on bounding boxes of street light. The bounding boxes are obtained in Step3.
Step5, Determine whether the answer is 'yes' or 'no' by executing Python expression, based on the number of street light. The number is obtained in Step4. If the number is greater than zero, the answer is 'yes'; On the contrary, the answer is 'no'.
Step6, Visualize results.
""",
"""Question: Is the log in front or behind the human in the center?
Subquestion: 
Step1, Locate the human in the center of the given image, and obtain bounding boxes of the animal.
Step2, Crop the image region in front of the human, based on bounding boxes of the human. The bounding boxes are obtained in Step1.
Step3, Crop the image region in front of the human, based on bounding boxes of the human. The bounding boxes are obtained in Step1.
Step4, Locate log in the image region in front of the human, and obtain bounding boxes of log. The image region in front of the human is cropped in Step2.
Step5, Count the number of log, based on bounding boxes of log. The bounding boxes are obtained in Step4.
Step6, Locate log in the image region in behind of the human, and obtain bounding boxes of log. The image region in front of the human is cropped in Step3.
Step7, Count the number of log, based on bounding boxes of log. The bounding boxes are obtained in Step5.
Step8, the question ask front or behind, so that determine whether the answer is 'front' or 'behind' by executing Python expression, based on the number of logs. The number is obtained in Step5 and Step 7 and. Based on the number, answer 'front' or 'behind' or 'neigher'.
Step9, Visualize results.
""",
"""Question: Are both the clouds and the pants the same color?
Subquestion: 
Step1, Locate the pants, and obtain bounding boxes of the pants.
Step2, Locate the clouds, and obtain bounding boxes of the clouds.
Step3, Crop the pants since the question requires knowing color of the pants. The bounding boxes are obtained in Step1.
Step4, Crop the clouds since the question requires knowing color of the clouds. The bounding boxes are obtained in Step2.
Step5, Ask image 'what color is the pants'. The image is cropped in Step3.
Step6, Ask image 'what color is the clouds'. The image is from Step4.
Step7, Determine whether colors are the same by executing Python expression.
Step6, Visualize results.
""",
"""Question: Is the car on the right side?
Subquestion: 
Step1, Locate the right side of the given image, and obtain bounding boxes.
Step2, Crop the right region based on bounding boxes obtained in Step1.
Step3, Locate the car of the cropped image from Step2.
Step4, Count the number of boxes from Step3.
Step5, Determine if the car is on the right side by Python expression. Answer yes if the result from Step4 are greater than 0.
Step6, Visualize results.
""",
"""Question: What color is the appliance above the bananas?
Subquestion: 
Step1, Locate the bananas, and obtain bounding boxes of the bananas.
Step2, Crop the above part of the bananas since the question is asking what is above the bananas. The bounding boxes are obtained in Step1.
Step3, Ask the image region, 'What color is it?' image. The image is cropped in Step2.
Step4, Visualize results.
""",
"""Question: What do both the pancake and the coffee mug have in common?
Subquestion: 
Step1, This question ask what two objects have in common without any detail information, therefore asking image region directly by the question 'What do both the pancake and the coffee mug have in common?'.
Step2, Visualize results.
""",
"""Question: How thick are the clouds the birds are flying in?
Subquestion: 
Step1, This question asks how thick are the clouds without any detail information, therefore asking image region directly by the question 'How thick are the clouds the birds are flying in?'.
Step2, Visualize results.
""",
"""Question: What material is the bath tub?
Subquestion: 
Step1, Locate the bath tub, and obtain bounding boxes of the bath tub.
Step2, Crop the bath tub since the question is asking what is the bath tub. The bounding boxes are obtained in Step1.
Step3, Ask the image region "what material is it?". The image is cropped in Step2.
Step4, Visualize results.
""",
"""Question: Who is standing?
Subquestion: 
Step1, This question asks who is standing without specifying location, species, and any other information, therefore asking image region directly by the question 'Who is standing?'.
Step2, Visualize results.
"""

]


CURATED_PROGRAMS=[
"""Question: Is the vehicle in the top of the image?
SubQuesion:
Step1, Locate the upper region of the given image since the question asks the top of the image, and obtain bounding boxes of the upper region.
Step2, Crop the upper region from the given image, based on bounding boxes of the upper region. The bounding boxes are obtained in Step1.
Step3, Locate vehicle in the upper region of the given image, and obtain bounding boxes of vehicle. The upper region is cropped in Step2.
Step4, Count the number of vehicle, based on bounding boxes of vehicle. The bounding boxes are obtained in Step3.
Step5, Determine whether the answer is 'yes' or 'no' by executing Python expression, based on the number of vehicles. The number is obtained in Step4. If the number is greater than zero, the answer is 'yes'; On the contrary, the answer is 'no'.
Step6, Visualize results.
Program:
BOX0=LOC(image=IMAGE,object='TOP')
IMAGE0=CROP(image=IMAGE,box=BOX0)
BOX1=LOC(image=IMAGE0,object='vehicle')
ANSWER0=COUNT(box=BOX1)
ANSWER1=EVAL(expr="'yes' if {ANSWER0} > 0 else 'no'")
FINAL_RESULT=RESULT(var=ANSWER1)
""",
"""Question: What type of candy is in the bowl that the pizza cutter is to the right of?
Subquestion: 
Step1, Locate the pizza cutter, and obtain bounding boxes of the pizza cutter.
Step2, Crop the left part of the pizza cutter since the question is asking what is to the right of the pizza cutter. The bounding boxes are obtained in Step1.
Step3, Since the question is asking what type, then ask the image region, 'What candy is it?' image. The image is cropped in Step2.
Step4, Visualize results.
Program:
BOX0=LOC(image=IMAGE,object='pizza cutter')
IMAGE0=CROP_LEFTOF(image=IMAGE,box=BOX0)
ANSWER0=VQA(image=IMAGE0, question='What candy is it?')
FINAL_RESULT=RESULT(var=ANSWER0)
""",
"""Question: Are the glass bowls to the left of a book?
Subquestion: 
Step1, Locate the book, and obtain bounding boxes of the book.
Step2, Crop the left part of the book since the question is asking what is to the left of the book. The bounding boxes are obtained in Step1.
Step3, Try locate glass bowls in the cropped image. The image is cropped in Step2.
Step4, Count the number of bounding boxes. The bounding box is from Step3.
Step5, This is a yes or no question, so determine whether the answer is 'yes' or 'no' by executing Python expression.
Step6, Visualize results.
Program:
BOX0=LOC(image=IMAGE,object='book')
IMAGE0=CROP_LEFTOF(image=IMAGE,box=BOX0)
BOX1=LOC(image=IMAGE0, object='glass bowls')
ANSWER0=COUNT(box=BOX1)
ANSWER1=EVAL(expr="'yes' if {ANSWER0} > 0 else 'no'")
FINAL_RESULT=RESULT(var=ANSWER1)
""",
"""Question: Does the cup that is to the right of the skateboarder look red?
Subquestion: 
Step1, Locate skateboarder in the given image, and obtain bounding boxes of skateboarder.
Step2, Crop the region right to the skateboarder, based on bounding boxes of skateboarder. The bounding boxes are obtained in Step1.
Step3, Asking the image region, 'What color is the cup?'. The image region of cup is obtained in Step2.
Step4, Determine whether the answer is 'yes' or 'no' by executing Python expression, based on the intermediate answers obtained in Step3. If the color is equal to 'red', the answer is 'yes'; On the contrary, the answer is 'no'.
Step5, Visualize results.
Program:
BOX0=LOC(image=IMAGE,object='skateboarder')
IMAGE0=CROP_RIGHTOF(image=IMAGE,box=BOX0)
ANSWER0=VQA(image=IMAGE0,question='What color is the cup?')
ANSWER1=EVAL(expr="'yes' if {ANSWER0} == 'red' else 'no'")
FINAL_RESULT=RESULT(var=ANSWER1)
""",
"""Question: What's on the table?
Subquestion: 
Step1, Locate the table, and obtain bounding boxes of the table.
Step2, Crop the table since the question is asking what is on the table. The bounding boxes are obtained in Step1.
Step3, Ask the image region "what's on the table?". The image is cropped in Step2.
Step4, Visualize results.
Program:
BOX0=LOC(image=IMAGE,object='table')
IMAGE0=CROP(image=IMAGE,box=BOX0)
ANSWER0=VQA(image=IMAGE0,question="what's on the table?")
FINAL_RESULT=RESULT(var=ANSWER0)
""",
"""Question: Is the street light standing behind a truck?
SubQuesion:
Step1, Locate truck in the given image, and obtain bounding boxes of truck.
Step2, Crop the image region behind the truck from the given image, based on bounding boxes of truck. The bounding boxes are obtained in Step1.
Step3, Locate street light in the region behind the truck, and obtain bounding boxes of street light. The region behind the truck is cropped in Step2.
Step4, Count the number of street light, based on bounding boxes of street light. The bounding boxes are obtained in Step3.
Step5, Determine whether the answer is 'yes' or 'no' by executing Python expression, based on the number of street light. The number is obtained in Step4. If the number is greater than zero, the answer is 'yes'; On the contrary, the answer is 'no'.
Step6, Visualize results.
Program:
BOX0=LOC(image=IMAGE,object='truck')
IMAGE0=CROP_BEHIND(image=IMAGE,box=BOX0)
BOX1=LOC(image=IMAGE0,object='street light')
ANSWER0=COUNT(box=BOX1)
ANSWER1=EVAL(expr="'yes' if {ANSWER0} > 0 else 'no'")
FINAL_RESULT=RESULT(var=ANSWER1)
""",
"""Question: Is the log in front or behind the human in the center?
Subquestion: 
Step1, Locate the human in the center of the given image, and obtain bounding boxes of the animal.
Step2, Crop the image region in front of the human, based on bounding boxes of the human. The bounding boxes are obtained in Step1.
Step3, Crop the image region in front of the human, based on bounding boxes of the human. The bounding boxes are obtained in Step1.
Step4, Locate log in the image region in front of the human, and obtain bounding boxes of log. The image region in front of the human is cropped in Step2.
Step5, Count the number of log, based on bounding boxes of log. The bounding boxes are obtained in Step4.
Step6, Locate log in the image region in behind of the human, and obtain bounding boxes of log. The image region in front of the human is cropped in Step3.
Step7, Count the number of log, based on bounding boxes of log. The bounding boxes are obtained in Step5.
Step8, the question ask front or behind, so that determine whether the answer is 'front' or 'behind' by executing Python expression, based on the number of logs. The number is obtained in Step5 and Step 7 and. Based on the number, answer 'front' or 'behind' or 'neigher'.
Step9, Visualize results.
Program:
BOX0=LOC(image=IMAGE,object='human')
IMAGE0=CROP_FRONTOF(image=IMAGE,box=BOX0)
IMAGE1=CROP_BEHIND(image=IMAGE,box=BOX0)
BOX1=LOC(image=IMAGE0,object='log')
ANSWER0=COUNT(box=BOX1)
BOX2=LOC(image=IMAGE1,object='log')
ANSWER1=COUNT(box=BOX2)
ANSWER2=EVAL(expr="'front' if {ANSWER0} > 0 else 'behind' if {ANSWER1} > 0 else 'neither'")
FINAL_RESULT=RESULT(var=ANSWER2)
""",
"""Question: Are both the clouds and the pants the same color?
Subquestion: 
Step1, Locate the pants, and obtain bounding boxes of the pants.
Step2, Locate the clouds, and obtain bounding boxes of the clouds.
Step3, Crop the pants since the question requires knowing color of the pants. The bounding boxes are obtained in Step1.
Step4, Crop the clouds since the question requires knowing color of the clouds. The bounding boxes are obtained in Step2.
Step5, Ask image 'what color is the pants'. The image is cropped in Step3.
Step6, Ask image 'what color is the clouds'. The image is from Step4.
Step7, Determine whether colors are the same by executing Python expression.
Step6, Visualize results.
Program:
BOX0=LOC(image=IMAGE,object='pants')
BOX1=LOC(image=IMAGE,object='clouds')
IMAGE0=CROP(image=IMAGE,box=BOX0)
IMAGE1=CROP(image=IMAGE,box=BOX1)
ANSWER0=VQA(image=IMAGE0,question='what color is the pants?')
ANSWER1=VQA(image=IMAGE1,question='what color is the clouds?')
ANSWER2=EVAL(expr="'yes' if {ANSWER0} == {ANSWER1} else 'no'")
FINAL_RESULT=RESULT(var=ANSWER2)
""",
"""Question: Is the car on the right side?
Subquestion: 
Step1, Locate the right side of the given image, and obtain bounding boxes.
Step2, Crop the right region based on bounding boxes obtained in Step1.
Step3, Locate the car of the cropped image from Step2.
Step4, Count the number of boxes from Step3.
Step5, Determine if the car is on the right side by Python expression. Answer yes if the result from Step4 are greater than 0.
Step6, Visualize results.
Program:
BOX0=LOC(image=IMAGE,object='RIGHT')
IMAGE0=CROP(image=IMAGE,box=BOX0)
BOX1=LOC(image=IMAGE0,object='car')
ANSWER0=COUNT(box=BOX1)
ANSWER1=EVAL(expr="'yes' if {ANSWER0} > 0 else 'no'")
FINAL_RESULT=RESULT(var=ANSWER1)
""",
"""Question: What color is the appliance above the bananas?
Subquestion: 
Step1, Locate the bananas, and obtain bounding boxes of the bananas.
Step2, Crop the above part of the bananas since the question is asking what is above the bananas. The bounding boxes are obtained in Step1.
Step3, Ask the image region, 'What color is it?' image. The image is cropped in Step2.
Step4, Visualize results.
Program:
BOX0=LOC(image=IMAGE,object='bananas')
IMAGE0=CROP_ABOVE(image=IMAGE,box=BOX0)
ANSWER0=VQA(image=IMAGE0, question='What color is it?')
FINAL_RESULT=RESULT(var=ANSWER0)
""",
"""Question: What do both the pancake and the coffee mug have in common?
Subquestion: 
Step1, This question ask what two objects have in common without any detail information, therefore asking image region directly by the question 'What do both the pancake and the coffee mug have in common?'.
Step2, Visualize results.
Program:
ANSWER0=VQA(image=IMAGE,question='What do both the pancake and the coffee mug have in common?')
FINAL_RESULT=RESULT(var=ANSWER0)
""","""Question: How thick are the clouds the birds are flying in?
Subquestion: 
Step1, This question asks how thick are the clouds without any detail information, therefore asking image region directly by the question 'How thick are the clouds the birds are flying in?'.
Step2, Visualize results.
Program:
ANSWER1=VQA(image=IMAGE,question='How thick are the clouds the birds are flying in?')
FINAL_RESULT=RESULT(var=ANSWER1)
""","""Question: What material is the bath tub?
Subquestion: 
Step1, Locate the bath tub, and obtain bounding boxes of the bath tub.
Step2, Crop the bath tub since the question is asking what is the bath tub. The bounding boxes are obtained in Step1.
Step3, Ask the image region "what material is it?". The image is cropped in Step2.
Step4, Visualize results.
Program:
BOX0=LOC(image=IMAGE,object='bath tub')
IMAGE0=CROP(image=IMAGE,box=BOX0)
ANSWER0=VQA(image=IMAGE0,question="what material is it?")
FINAL_RESULT=RESULT(var=ANSWER0)
""",
"""Question: Who is standing?
Subquestion: 
Step1, This question asks who is standing without specifying location, species, and any other information, therefore asking image region directly by the question 'Who is standing?'.
Step2, Visualize results.
Program:
ANSWER1=VQA(image=IMAGE,question='Who is standing?')
FINAL_RESULT=RESULT(var=ANSWER1)
"""

]


REFLECTION_STEP=[
"""Question: What's the window made of?
Description of the Input Image: a photography of a cat sitting on a table watching tv
Human Feedback: the correct answer should be glass
Our Wrong Answer: metal
subQuesion:
Step1, Locate window in the given image, and obtain bounding boxes of window.
Step2, Crop the region of window from the given image, based on bounding boxes of window. The bounding boxes are obtained in Step1.
Step3, Asking the image region of window, 'What's the window made of?'. The image region of window is obtained in Step2.
Step4, Visualize results.
Program and obtained result in each step:
Step1
Program: BOX0=LOC(image=IMAGE,object='window')
Result of BOX0 is empty 
Step2
Program: IMAGE0=CROP(image=IMAGE,box=BOX0)
Result of The description of IMAGE0: a photography of a cat sitting on a table watching tv
Step3
Program: ANSWER0=VQA(image=IMAGE0,question='What's the window made of?')
Result of ANSWER0: cat
Step4
Program: FINAL_RESULT=RESULT(var=ANSWER0)
Result of FINAL_RESULT: metal
Error Location: functions called by programs
Reason: In the Step1 of the program, the used function 'LOC' failed to locate the window in the given image, as the obtained result of BOX0 is empty.
""",
"""Question: Is the lamp different in color than the shirt?
Description of the Input Image: a photography of a couple of people on a snowboard in the snow
Human Feedback: the correct answer should be yes
Our Wrong Answer: no
subquestion: 
Step1, Locate the lamp in the given image, and obtain bounding boxes of lamp.
Step2, Crop the region of the lamp from the given image, based on bounding boxes of lamp. The bounding boxes are obtained in Step1.
Step3, Asking the image region of lamp, 'What color is the lamp?'. The image region of lamp is cropped in Step2.
Step4, Locate the shirt in the given image, and obtain bounding boxes of shirt. 
Step5, Crop the region of the shirt from the given image, based on bounding boxes of shirt. The bounding boxes are obtained in Step4.
Step6, Asking the image region of shirt, 'What color is the shirt?'. The image region of lamp is cropped in Step5.
Step7, Determine whether the answer is 'yes' or 'no' by executing Python expression, based on the color of lamp and the color of shirt. The color of lamp and shirt is obtained in Step3 and Step6, respectively. If their color are the same, the answer is 'yes'; On the contrary, the answer is 'no'.
Step8, Visualize results.
Program and obtained result in each step:
Step1
Program: BOX0=LOC(image=IMAGE,object='lamp')
Result of The coordinate of BOX0: [[45, 78, 245, 345]]
Step2
Program: IMAGE0=CROP(image=IMAGE,box=BOX0)
Result of The description of IMAGE0: a photography of a couple of people on a snowboard in the snow
Step3
Program: BOX1=LOC(image=IMAGE0,object='shirt')
Result of BOX1 is empty 
Step4
Program: ANSWER0=COUNT(box=BOX1)
Result of ANSWER0: 0
Step5
Program: ANSWER1=EVAL(expr="'yes' if {ANSWER0} > 0 else 'no'")
Result of ANSWER1: no
Step6
Program: FINAL_RESULT=RESULT(var=ANSWER1)
Result of FINAL_RESULT: no
Error Location: program
Reason: The subquestions are correct, and can address the given question. But the Step3-Step6 of the program does not match the subquestion. The subquestions locate, crop, and ask color of the lamp and shirt, but the program counts the number of lamp.
""",
"""Question: Is the pipe made of the same material as the bed sheet?
Description of the Input Image: a photography of a kitchen with a refrigerator and a stove
Human Feedback: the correct answer should be yes
Our Wrong Answer: metal
subquestion: 
Step1, Locate pipe in the given image, and obtain bounding boxes of pipe.
Step2, Crop the region of pipe from the given image, based on bounding boxes of pipe. The bounding boxes are obtained in Step1.
Step3, Asking the image region of pipe, 'What material is the pipe made of?'. The image region of pipe is obtained in Step2.
Step4, Visualize results.
Program and obtained result in each step:
Step1
Program: BOX0=LOC(image=IMAGE,object='pipe')
Result of The coordinate of BOX0: [[67, 28, 98, 69]]
Step2
Program: IMAGE0=CROP(image=IMAGE,box=BOX0)
Result of The description of IMAGE0: a photography of a man is holding a knife in his hand
Step3
Program: ANSWER0=VQA(image=IMAGE0,question='What material is the pipe made of?')
Result of ANSWER0: metal
Step4
Program: FINAL_RESULT=RESULT(var=ANSWER0)
Result of FINAL_RESULT: metal
Error Location: subquestions
Reason: In Step1-Step3, the subquestions only identify the material of the pipe. The subquestions should then to identify the material of the bed sheet, and then compare the material of the pipe and the bed sheet.
""",
"""Question: What type of animal is to the left of the people?
Description of the Input Image: a photography of a group of giraffes and zebras in a field
Human Feedback: the correct answer should be zebras
Our Wrong Answer: dog
subquestion: 
Step1, Locate people in the given image, and obtain bounding boxes of people.
Step2, Crop the region of people from the given image, based on bounding boxes of people. The bounding boxes are obtained in Step1.
Step3, Asking the image region of people, 'What type of animal is to the left of the people?'. The image region of people is obtained in Step2.
Step4, Visualize results.
Program and obtained result in each step:
Step1
Program: BOX0=LOC(image=IMAGE,object='people')
Result of The coordinate of BOX0: [[545, 241, 562, 266], [592, 245, 606, 279]]
Step2
Program: IMAGE0=CROP(image=IMAGE,box=BOX0)
Result of The description of IMAGE0: a photography of a man standing next to a white refrigerator
Step3
Program: ANSWER0=VQA(image=IMAGE0,question='What type of animal is to the left of the people?')
Result of ANSWER0: dog
Step4
Program: FINAL_RESULT=RESULT(var=ANSWER0)
Result of FINAL_RESULT: dog
Error Location: subquestions
Reason: In Step2 of the subquestions, the subquestions should crop the image region on the left side of people, instead of cropping the region of people, since the question asks 'What type of animal is to the left of the people?'.
""",
"""Question: Is the water dark and wet?
Description of the Input Image: a photography of a woman walking down a street holding an umbrella
Human Feedback: the correct answer should be yes
Our Wrong Answer: no
subquestion: 
Step1, Locate water in the given image, and obtain bounding boxes of water.
Step2, Crop the region of water from the given image, based on bounding boxes of water. The bounding boxes are obtained in Step1.
Step3, Asking the image region of water, 'Is the water dark?'. The image region of water is obtained in Step2.
Step4, Asking the image region of water, 'Is the water wet?'. The image region of water is obtained in Step2.
Step5, Determine whether the answer is 'yes' or 'no' by executing Python expression, based on the intermediate answers obtained in Step3 and Step4. If the answer of Step3 is 'yes' and the answer of Step4 is 'yes', the answer is 'yes'; On the contrary, the answer is 'no'.
Step6, Visualize results.
Program and obtained result in each step:
Step1
Program: BOX0=LOC(image=IMAGE,object='water')
Result of The coordinate of BOX0: [[136, 45, 606, 279]]
Step2
Program: IMAGE0=CROP(image=IMAGE,box=BOX0)
Result of The description of IMAGE0: a photography of a street
Step3
Program: ANSWER0=VQA(image=IMAGE0,question='Is the water dark?')
Result of ANSWER0: no
Step4
Program: ANSWER1=VQA(image=IMAGE0,question='Is the water wet?')
Result of ANSWER1: yes
Step5
Program: ANSWER2=EVAL(expr="'yes' if {ANSWER0} and {ANSWER1} else 'no'")
Result of ANSWER2: no
Step6
Program: FINAL_RESULT=RESULT(var=ANSWER2)
Result of FINAL_RESULT: no
Error Location: functions called by programs
Reason: In the Step3 of the program, the used function 'VQA' failed failed to identify whether the water is dark correctly, as the obtained result of ANSWER0 is 'no' instead of 'yes'.
""",
"""Question: Are there either any windows or trains in this image?
Description of the Input Image: a photography of a woman and a man playing a video game
Human Feedback: the correct answer should be yes
Our Wrong Answer: no
subquestion: 
Step1, Locate any windows in the given image, and obtain bounding boxes of any windows.
Step2, Crop the region of any windows from the given image, based on bounding boxes of any windows. The bounding boxes are obtained in Step1.
Step3, Asking the image region of any windows, 'Are there either any windows or trains in this image?'. The image region of any windows is obtained in Step2.
Step4, Determine whether the answer is 'yes' or 'no' by executing Python expression, based on the intermediate answers obtained in Step3. If the answer is 'yes', the answer is 'yes'; On the contrary, the answer is 'no'.
Step5, Visualize results.
Program and obtained result in each step:
Step1
Program: BOX0=LOC(image=IMAGE,object='window')
Result of The coordinate of BOX0: [[274, 93, 408, 327], [25, 59, 192, 331]]
Step2
Program: IMAGE0=CROP(image=IMAGE,box=BOX0)
Result of The description of IMAGE0: a photography of a man in a plaid shirt is holding a wii controller
Step3
Program: ANSWER0=VQA(image=IMAGE0,question='Are there either any windows or trains in this image?')
Result of ANSWER0: yes
Step4
Program: ANSWER1=EVAL(expr="'yes' if {ANSWER0} == 'yes' else 'no'")
Result of ANSWER1: no
Step5
Program: FINAL_RESULT=RESULT(var=ANSWER1)
Result of FINAL_RESULT: no
Error Location: subquestions
Reason: In Step2 of the subquestions, the subquestions should count the number of boxes of windows', instead of cropping the image and asking question. Then, the subquestions should further detect trains and count the number of trains.
""",
# """Question: Do you think the table is rectangular?
# The description of Input image: a photography of a restaurant with a table set for a meal
# Human Feedback: the correct answer should be yes
# Our Wrong Answer: no
# Following are the decomposed subquestion, used program, and obtained result in each step.
# subquestion:
# Step1, Locate table in the given image, and obtain bounding boxes of table.
# Step2, Crop the region of table from the given image, based on bounding boxes of table. The bounding boxes are obtained in Step1.
# Step3, Asking the image region of table, 'What shape is the table?'. The image region of table is obtained in Step2.
# Step4, Determine whether the answer is 'yes' or 'no' by executing Python expression, based on the intermediate answers obtained in Step3. If the shape is equal to 'rectangular', the answer is 'yes'; On the contrary, the answer is 'no'.
# Step5, Visualize results.
# Program and obtained result in each step:
# Step1
# Program: BOX0=LOC(image=IMAGE,object='table')
# The coordinate of BOX0: [[40, 240, 581, 479], [180, 200, 258, 243]]
# Step2
# Program: IMAGE0=CROP(image=IMAGE,box=BOX0)
# The description of IMAGE0: a photography of a table set with a white table cloth and red plates
# Step3
# Program: ANSWER0=VQA(image=IMAGE0,question='What shape is the table?')
# Results of ANSWER0: circular
# Step4
# Program: ANSWER1=EVAL(expr="'yes' if {ANSWER0} == 'rectangular' else 'no'")
# Result of ANSWER1: no
# Step5
# Program: INAL_RESULT=RESULT(var=ANSWER1)
# Result of FINAL_RESULT: no
# Error Location: functions called by programs
# Reason: In the Step3 of the program, the used function 'VQA' failed to recognize the shape of the table correctly, as the obtained result of ANSWER0 is 'rectangle' instead of 'circular'.
# """,
]


REFLECTION_INTERRUPT=[
"""Question: Is there a plate to the left of the food tray in the top?
SubQuesion:
Step1, Locate food tray in the given image, and obtain bounding boxes of food tray.
Step2, Crop the image region on the left side of food tray from the given image, based on bounding boxes of food tray. The bounding boxes are obtained in Step1.
Step3, Locate plate in the image region on the left side of the food tray, and obtain bounding boxes of plate. The image region on the left side of the food tray is cropped in Step2.
Step4, Count the number of plate, based on bounding boxes of plate. The bounding boxes are obtained in Step3.
Step5, Determine whether the answer is 'yes' or 'no' by executing Python expression, based on the number of plate. The number is obtained in Step4. If the number is greater than zero, the answer is 'yes'; On the contrary, the answer is 'no'.
Step6, Visualize results.
Program: 
BOX0=LOC(image=IMAGE,object='food tray')
IMAGE0=CROP_LEFT(image=IMAGE,box=BOX0)
BOX1=LOC(image=IMAGE0,object='plate')
ANSWER0=COUNT(box=BOX1)
ANSWER1=EVAL(expr="'yes' if {ANSWER0} > 0 else 'no'")
FINAL_RESULT=RESULT(var=ANSWER1)
Error Location: 
Second line of the program: IMAGE0=CROP_LEFT(image=IMAGE,box=BOX0)
Reason: 
The bug in the program is in the second line: IMAGE0=CROP_LEFT(image=IMAGE,box=BOX0), where the function 'CROP_LEFT' is called. It should be 'CROP_LEFTOF'.
""",
"""Question: Is the vehicle in the top of the image?
SubQuesion:
Step1, Locate the upper region of the given image, and obtain bounding boxes of the upper region.
Step2, Crop the upper region from the given image, based on bounding boxes of the upper region. The bounding boxes are obtained in Step1.
Step3, Locate vehicle in the upper region of the given image, and obtain bounding boxes of vehicle. The upper region is cropped in Step2.
Step4, Count the number of vehicle, based on bounding boxes of vehicle. The bounding boxes are obtained in Step3.
Step5, Determine whether the answer is 'yes' or 'no' by executing Python expression, based on the number of vehicles. The number is obtained in Step4. If the number is greater than zero, the answer is 'yes'; On the contrary, the answer is 'no'.
Step6, Visualize results.
Program:
BOX0=LOC(image=IMAGE,object='TOP')
IMAGE0=CROP(image=IMAGE,box=BOX0)
BOX1=LOC(image=IMAGE1,object='vehicle')
ANSWER0=COUNT(box=BOX1)
ANSWER1=EVAL(expr="'yes' if {ANSWER0} > 0 else 'no'")
FINAL_RESULT=RESULT(var=ANSWER1)
Error Location: 
Third line of the program: BOX1=LOC(image=IMAGE1,object='vehicle')
Reason:
The bug in the program is in the third line: BOX1=LOC(image=IMAGE1,object='vehicle'), where the variable 'IMAGE1' is called. It should be 'IMAGE'.
""",
]


INFERENCE=[
"""Question: Do you think the table is rectangular?
The description of Input image: a photography of a restaurant with a table set for a meal
Human Feedback: the correct answer should be yes
Our Wrong Answer: no
Following are the decomposed subquestion, used program, and obtained result in each step. 
subquestion: 
Step1, Locate table in the given image, and obtain bounding boxes of table.
Step2, Crop the region of table from the given image, based on bounding boxes of table. The bounding boxes are obtained in Step1.
Step3, Asking the image region of table, 'What shape is the table?'. The image region of table is obtained in Step2.
Step4, Determine whether the answer is 'yes' or 'no' by executing Python expression, based on the intermediate answers obtained in Step3. If the shape is equal to 'rectangular', the answer is 'yes'; On the contrary, the answer is 'no'.
Step5, Visualize results.
Program and obtained result in each step:
Step1
Program: BOX0=LOC(image=IMAGE,object='table')
The coordinate of BOX0: [[40, 240, 581, 479], [180, 200, 258, 243]]
Step2
Program: IMAGE0=CROP(image=IMAGE,box=BOX0)
The description of IMAGE0: a photography of a table set with a white table cloth and red plates
Step3
Program: ANSWER0=VQA(image=IMAGE0,question='What shape is the table?')
Results of ANSWER0: circular
Step4
Program: ANSWER1=EVAL(expr="'yes' if {ANSWER0} == 'rectangular' else 'no'")
Result of ANSWER1: no
Step5
Program: INAL_RESULT=RESULT(var=ANSWER1)
Result of FINAL_RESULT: no
Error Location: functions called by programs
Reason: In the Step3 of the program, the used function 'VQA' failed to recognize the shape of the table correctly, as the obtained result of ANSWER0 is 'rectangle' instead of 'circular'.
Correct answer of the wrong step: rectangle.
""",
]

# 根据错误原因重新生成子步骤的经验池
FAILED_SUBQUESTION_WITH_REFLECTION=[
"""
Question: Is the dog to the right of the chair both brown and small?
Initial planned sub-steps:
Step1, Locate the chair, and obtain bounding boxes of the chair.
Step2, Crop the right part of the chair since the question is asking what is to the right of the chair. The bounding boxes are obtained in Step1.
Step3, Try locate dog in the cropped image. The image is cropped in Step2.
Step4, Count the number of bounding boxes. The bounding box is from Step3.
Step5, This is a yes or no question, so determine whether the answer is 'yes' or 'no' by executing Python expression.
Step6, Visualize results.
Initial generated program:
BOX0=LOC(image=IMAGE,object='chair')
IMAGE0=CROP_RIGHTOF(image=IMAGE,box=BOX0)
BOX1=LOC(image=IMAGE0,object='dog')
BOX2=LOC(image=IMAGE0,object='dog')
ANSWER0=COUNT(box=BOX1)
ANSWER1=COUNT(box=BOX2)
ANSWER2=EVAL(expr="'yes' if {{ANSWER0}} > 0 and {{ANSWER1}} > 0 else 'no'")
FINAL_RESULT=RESULT(var=ANSWER2)
Error reason:
The old plan does not actually verify whether the dog is both brown and small. It only repeats dog detection and then uses count-based yes/no logic, so it ignores the required attributes.
Revised subquestion:
Step1, Locate the chair, and obtain bounding boxes of the chair.
Step2, Crop the image region to the right of the chair, based on bounding boxes obtained in Step1.
Step3, Ask whether the dog in the cropped image region from Step2 is both brown and small.
Step4, Visualize results.
""",
"""
Question: What material is the bag of the woman on the left?
Initial planned sub-steps:
Step1, Locate the woman, and obtain bounding boxes of the woman.
Step2, Crop the image region of the woman, based on bounding boxes obtained in Step1.
Step3, Locate the bag in the cropped image region, and obtain bounding boxes of the bag.
Step4, Count the number of bag bounding boxes, based on Step3.
Step5, Determine whether the answer is 'leather' or 'cloth' by executing Python expression, based on the number from Step4.
Step6, Visualize results.
Initial generated program:
BOX0=LOC(image=IMAGE,object='woman')
IMAGE0=CROP(image=IMAGE,box=BOX0)
BOX1=LOC(image=IMAGE0,object='bag')
ANSWER0=COUNT(box=BOX1)
ANSWER1=EVAL(expr="'leather' if {{ANSWER0}} > 0 else 'cloth'")
FINAL_RESULT=RESULT(var=ANSWER1)
Error reason:
The old plan changes a material question into count-based fixed-choice logic, and it ignores the important constraint 'on the left'.
Revised subquestion:
Step1, Locate the woman on the left, and obtain bounding boxes of the woman.
Step2, Crop the image region of the woman, based on bounding boxes obtained in Step1.
Step3, Ask what material the bag is in the cropped image region from Step2.
Step4, Visualize results.
""",
]
# 根据错误原因重新生成程序的经验池
FAILED_PROGRAM_WITH_REFLECTION=[
"""
Question: Is the dog to the right of the chair both brown and small?
Initial planned sub-steps:
Step1, Locate the chair, and obtain bounding boxes of the chair.
Step2, Crop the right part of the chair since the question is asking what is to the right of the chair. The bounding boxes are obtained in Step1.
Step3, Try locate dog in the cropped image. The image is cropped in Step2.
Step4, Count the number of bounding boxes. The bounding box is from Step3.
Step5, This is a yes or no question, so determine whether the answer is 'yes' or 'no' by executing Python expression.
Step6, Visualize results.
Initial generated program:
BOX0=LOC(image=IMAGE,object='chair')
IMAGE0=CROP_RIGHTOF(image=IMAGE,box=BOX0)
BOX1=LOC(image=IMAGE0,object='dog')
BOX2=LOC(image=IMAGE0,object='dog')
ANSWER0=COUNT(box=BOX1)
ANSWER1=COUNT(box=BOX2)
ANSWER2=EVAL(expr="'yes' if {{ANSWER0}} > 0 and {{ANSWER1}} > 0 else 'no'")
FINAL_RESULT=RESULT(var=ANSWER2)
Error reason:
The old plan does not actually verify whether the dog is both brown and small. It only repeats dog detection and then uses count-based yes/no logic, so it ignores the required attributes.
Revised subquestion:
Step1, Locate the chair, and obtain bounding boxes of the chair.
Step2, Crop the image region to the right of the chair, based on bounding boxes obtained in Step1.
Step3, Ask whether the dog in the cropped image region from Step2 is both brown and small.
Step4, Visualize results.
Revised program:
BOX0=LOC(image=IMAGE,object='chair')
IMAGE0=CROP_RIGHTOF(image=IMAGE,box=BOX0)
ANSWER0=VQA(image=IMAGE0,question='Is the dog both brown and small?')
FINAL_RESULT=RESULT(var=ANSWER0)
""",
"""
Question: What material is the bag of the woman on the left?
Initial planned sub-steps:
Step1, Locate the woman, and obtain bounding boxes of the woman.
Step2, Crop the image region of the woman, based on bounding boxes obtained in Step1.
Step3, Locate the bag in the cropped image region, and obtain bounding boxes of the bag.
Step4, Count the number of bag bounding boxes, based on Step3.
Step5, Determine whether the answer is 'leather' or 'cloth' by executing Python expression, based on the number from Step4.
Step6, Visualize results.
Initial generated program:
BOX0=LOC(image=IMAGE,object='woman')
IMAGE0=CROP(image=IMAGE,box=BOX0)
BOX1=LOC(image=IMAGE0,object='bag')
ANSWER0=COUNT(box=BOX1)
ANSWER1=EVAL(expr="'leather' if {{ANSWER0}} > 0 else 'cloth'")
FINAL_RESULT=RESULT(var=ANSWER1)
Error reason:
The old plan changes a material question into count-based fixed-choice logic, and it ignores the important constraint 'on the left'.
Revised subquestion:
Step1, Locate the woman on the left, and obtain bounding boxes of the woman.
Step2, Crop the image region of the woman, based on bounding boxes obtained in Step1.
Step3, Ask what material the bag is in the cropped image region from Step2.
Step4, Visualize results.
Revised program:
BOX0=LOC(image=IMAGE,object='woman on the left')
IMAGE0=CROP(image=IMAGE,box=BOX0)
ANSWER0=VQA(image=IMAGE0,question='What material is the bag?')
FINAL_RESULT=RESULT(var=ANSWER0)
""",
]
# 生成出错原因的经验池
EXPERIENCE_POOL_WITH_REFLECTION_RESULT= [
"""
Input:
1. User request:
What brand is the white sneaker on the top shelf?

2. Planned sub-steps:
Step1, Locate the sneaker, and obtain bounding boxes of the sneaker.
Step2, Crop the image region of the sneaker based on bounding boxes obtained in Step1.
Step3, Locate the brand logo in the cropped image.
Step4, Count the number of bounding boxes.
Step5, Determine whether the answer is 'Nike' or 'Adidas' by executing Python expression.
Step6, Visualize results.

3. Generated code:
BOX0=LOC(image=IMAGE,object='sneaker')
IMAGE0=CROP(image=IMAGE,box=BOX0)
BOX1=LOC(image=IMAGE0,object='brand logo')
ANSWER0=COUNT(box=BOX1)
ANSWER1=EVAL(expr="'Nike' if {ANSWER0} > 0 else 'Adidas'")
FINAL_RESULT=RESULT(var=ANSWER1)

Output:
{
  "error_type": "PLAN_STEP_DECOMPOSITION_ERROR, API_PARAMETER_ERROR",
  "error_layer": "TASK_PLANNING",
  "substep_ids": ["Step1", "Step4", "Step5"],
  "error_reason": "In Step 1, the original question explicitly stated 'the white sneaker on the top shelf', but the LOC tool only used 'sneaker', completely omitting the crucial color and spatial attributes. If there are multiple sneakers, the tool will target the wrong one. Furthermore, the original question is 'What brand...', which is an open-ended extraction task. In Steps 4 and 5, the plan forces this into a closed binary guess using COUNT and EVAL (assuming >0 means 'Nike'), resulting in an arbitrary hallucination that completely deviates from the user's intention.",
  "fix_strategy": "Step 1: Add the attributes 'white' and 'on the top shelf' to the LOC tool parameter. Steps 3, 4, and 5: Abandon LOC, COUNT, and EVAL for text/brand extraction. Pass the cropped sneaker directly to the VQA tool and ask 'What brand is this sneaker?'."
}
""",
"""
Input:
1. User request:
Is the cat sleeping on the rug next to the fireplace?

2. Planned sub-steps:
Step1, Locate the fireplace, and obtain bounding boxes.
Step2, Crop the region right to the fireplace since the question asks what is next to it.
Step3, Locate the cat in the cropped image.
Step4, Count the number of bounding boxes.
Step5, Use EVAL to return 'yes' if count is greater than 0, else 'no'.
Step6, Visualize results.

3. Generated code:
BOX0=LOC(image=IMAGE,object='fireplace')
IMAGE0=CROP_RIGHTOF(image=IMAGE,box=BOX0)
BOX1=LOC(image=IMAGE0,object='cat')
ANSWER0=COUNT(box=BOX1)
ANSWER1=EVAL(expr="'yes' if {ANSWER0} > 0 else 'no'")
FINAL_RESULT=RESULT(var=ANSWER1)

Output:
{
  "error_type": "PLAN_STEP_DECOMPOSITION_ERROR",
  "error_layer": "TASK_PLANNING",
  "substep_ids": ["Step2", "Step3"],
  "error_reason": "In Step 2, the plan makes a false spatial assumption by rigidly converting 'next to' into CROP_RIGHTOF. The cat could be on the left or in front of the fireplace, meaning the crop will likely miss the subject entirely. In Step 3, the LOC tool searches for 'cat' but completely omits the critical state 'sleeping on the rug'. Finding a standing cat not on a rug would still trigger a 'yes' in Step 5, violating the complex conditional requirements of the user's prompt.",
  "fix_strategy": "Do not decompose ambiguous relational constraints like 'next to' into rigid geometric crops. Maintain the original image context. Submit the uncropped image to the VQA tool and ask the full question 'Is the cat sleeping on the rug next to the fireplace?' to evaluate all constraints simultaneously."
}
""",
"""
Input:
1. User request:
How many people are waiting at the bus stop across the street?

2. Planned sub-steps:
Step1, Locate the bus stop across the street, and obtain bounding boxes.
Step2, Crop the image region of the bus stop.
Step3, Ask the VQA tool 'how many people are waiting?'.
Step4, Visualize results.

3. Generated code:
BOX0=LOC(image=IMAGE,object='bus stop across the street')
IMAGE0=CROP(image=IMAGE,box=BOX0)
ANSWER0=VQA(image=IMAGE0,question='how many people are waiting?')
FINAL_RESULT=RESULT(var=ANSWER0)

Output:
{
  "error_type": "TOOL_CAPABILITY_LIMIT",
  "error_layer": "EXECUTION",
  "substep_ids": ["Step3"],
  "error_reason": "In Step 3, the plan delegates an exact numerical counting task ('How many people...') to the VQA tool. VQA models are semantic feature extractors and are notoriously unreliable at performing exact mathematical tallies on multiple objects, often hallucinating a random number when the count exceeds 3 or 4. This tool choice leads to highly inaccurate statistics.",
  "fix_strategy": "For precise numerical counting, replace VQA with a detection-based pipeline. In Step 3, use the LOC tool to target 'people' within the cropped bus stop region, then add a Step 4 using the COUNT tool to calculate the exact number of resulting bounding boxes."
}
""",
"""
Input:
1. User request:
What text is written on the billboard above the highway?

2. Planned sub-steps:
Step1, Locate the highway.
Step2, Crop the region above the highway.
Step3, Locate the text in the cropped image.
Step4, Crop the text region.
Step5, Ask the VQA tool 'what is written here?'.

3. Generated code:
BOX0=LOC(image=IMAGE,object='highway')
IMAGE0=CROP_ABOVE(image=IMAGE,box=BOX0)
BOX1=LOC(image=IMAGE0,object='text')
IMAGE1=CROP(image=IMAGE0,box=BOX1)
ANSWER0=VQA(image=IMAGE1,question='what is written here?')
FINAL_RESULT=RESULT(var=ANSWER0)

Output:
{
  "error_type": "PLAN_STEP_DECOMPOSITION_ERROR, OCR_CONTEXT_STRIPPING",
  "error_layer": "TASK_PLANNING",
  "substep_ids": ["Step3", "Step4"],
  "error_reason": "In Step 3, the LOC tool is incorrectly tasked with finding 'text'. Object detection models are designed for physical nouns, not semantic strings or OCR grouping. Furthermore, in Step 4, executing a tight crop around hallucinated 'text' bounding boxes strips away the visual background of the billboard, destroying the resolution and context required for the VQA tool to successfully read the characters in Step 5.",
  "fix_strategy": "Skip the secondary LOC and CROP tools intended to isolate text. Pass the broader spatial crop (IMAGE0, the region above the highway containing the billboard) directly to the VQA tool and ask it to read the text."
}
""",
"""
Input:
1. User request:
Is the window of the blue car rolled down?

2. Planned sub-steps:
Step1, Locate the blue car, and obtain bounding boxes.
Step2, Crop the blue car region.
Step3, Locate the rolled down window in the cropped image.
Step4, Count the bounding boxes.
Step5, Output yes if count > 0, else no.

3. Generated code:
BOX0=LOC(image=IMAGE,object='blue car')
IMAGE0=CROP(image=IMAGE,box=BOX0)
BOX1=LOC(image=IMAGE0,object='rolled down window')
ANSWER0=COUNT(box=BOX1)
ANSWER1=EVAL(expr="'yes' if {ANSWER0} > 0 else 'no'")
FINAL_RESULT=RESULT(var=ANSWER1)

Output:
{
  "error_type": "PLAN_STEP_DECOMPOSITION_ERROR",
  "error_layer": "TASK_PLANNING",
  "substep_ids": ["Step3", "Step4", "Step5"],
  "error_reason": "The original question requires verifying a complex physical state ('rolled down'). In Step 3, the plan mistakenly uses the LOC tool to detect this state. LOC models prioritize object existence over fine-grained state variations. If the tool detects the window but it is actually closed, the bounding box count will still be greater than 0, causing the EVAL logic in Step 5 to erroneously output 'yes'.",
  "fix_strategy": "Do not dismantle physical state verification queries with LOC and COUNT combinations. Pass the cropped image of the blue car (IMAGE0) directly to the VQA tool and ask the explicit boolean question 'Is the window of the car rolled down?'."
}
""",
"""
Input:
1. User request:
Is the red apple larger than the green apple?

2. Planned sub-steps:
Step1, Locate the red apple, and crop it.
Step2, Locate the green apple, and crop it.
Step3, Ask VQA on the first crop 'Is it larger?'.
Step4, Output result.

3. Generated code:
BOX0=LOC(image=IMAGE,object='red apple')
IMAGE0=CROP(image=IMAGE,box=BOX0)
BOX1=LOC(image=IMAGE,object='green apple')
IMAGE1=CROP(image=IMAGE,box=BOX1)
ANSWER0=VQA(image=IMAGE0,question='Is it larger?')
FINAL_RESULT=RESULT(var=ANSWER0)

Output:
{
  "error_type": "PLAN_STEP_DECOMPOSITION_ERROR, CROSS_SPATIAL_COMPARISON_DESTRUCTION",
  "error_layer": "TASK_PLANNING",
  "substep_ids": ["Step1", "Step2", "Step3"],
  "error_reason": "The user is asking for a relative size comparison ('larger than') between two objects. By isolating the apples into two separate crops in Steps 1 and 2, the plan permanently destroys the shared baseline coordinate system and visual context. In Step 3, asking a VQA model 'Is it larger?' on a single, isolated image crop is a logical paradox, as the model has absolutely no reference object against which to compare it.",
  "fix_strategy": "Relative size comparisons must be evaluated holistically. Do not separate the objects into individual crops. Provide the single, original, uncropped image to the VQA tool and directly ask 'Is the red apple larger than the green apple?'."
}
""",
"""
Input:
1. User request:
Does the shadow of the building touch the parked bicycle?

2. Planned sub-steps:
Step1, Locate the shadow of the building.
Step2, Locate the parked bicycle.
Step3, Ask VQA if the boxes touch each other.
Step4, Visualize.

3. Generated code:
BOX0=LOC(image=IMAGE,object='shadow of the building')
BOX1=LOC(image=IMAGE,object='parked bicycle')
ANSWER0=VQA(image=IMAGE,question='Do the items in BOX0 and BOX1 touch each other?')
FINAL_RESULT=RESULT(var=ANSWER0)

Output:
{
  "error_type": "PLAN_STEP_DECOMPOSITION_ERROR",
  "error_layer": "TASK_PLANNING",
  "substep_ids": ["Step1", "Step3"],
  "error_reason": "In Step 1, the plan attempts to use LOC to find a 'shadow'. Shadows are phenomenological lighting effects, not bounded physical objects; LOC tools will likely fail or output arbitrary boxes. In Step 3, the plan commits a severe API misuse by passing internal code variables ('BOX0' and 'BOX1') directly into the natural language prompt of the VQA tool. VQA cannot interpret these internal variable references.",
  "fix_strategy": "Do not attempt to pass code variables into VQA prompts, and do not use LOC for shadows or lighting. To evaluate geometric intersections or environmental lighting interactions, submit the unmodified image to the VQA model and ask 'Does the shadow of the building touch the parked bicycle?'."
}
""",
"""
Input:
1. User request:
Is the man wearing a hat holding a coffee cup?

2. Planned sub-steps:
Step1, Locate the man, and crop the region.
Step2, Locate the coffee cup in the cropped image.
Step3, Count the bounding boxes.
Step4, Use EVAL to return yes if count > 0.

3. Generated code:
BOX0=LOC(image=IMAGE,object='man')
IMAGE0=CROP(image=IMAGE,box=BOX0)
BOX1=LOC(image=IMAGE0,object='coffee cup')
ANSWER0=COUNT(box=BOX1)
ANSWER1=EVAL(expr="'yes' if {ANSWER0} > 0 else 'no'")
FINAL_RESULT=RESULT(var=ANSWER1)

Output:
{
  "error_type": "PLAN_STEP_DECOMPOSITION_ERROR",
  "error_layer": "TASK_PLANNING",
  "substep_ids": ["Step1", "Step2", "Step4"],
  "error_reason": "In Step 1, the LOC tool omits the crucial identifying attribute 'wearing a hat', risking tracking the wrong person. More critically, the question requires verifying a kinematic interaction ('holding'). Steps 2 and 4 downgrade this action verification into a mere existence check for a 'coffee cup' inside the crop. A coffee cup sitting on a table in the background of the man's crop would falsely trigger a 'yes' output, breaking causal logic.",
  "fix_strategy": "Step 1: Add 'wearing a hat' to the LOC parameter. Step 2 & 4: Remove LOC, COUNT, and EVAL for action verification. Pass the cropped image of the correct man to the VQA tool and ask explicitly, 'Is he holding a coffee cup?'."
}
""",
"""
Input:
1. User request:
What is the material of the coat the woman on the right is wearing?

2. Planned sub-steps:
Step1, Locate the woman, and crop the region.
Step2, Locate the coat.
Step3, Count the boxes and evaluate.
Step4, Output 'leather' if >0 else 'wool'.

3. Generated code:
BOX0=LOC(image=IMAGE,object='woman')
IMAGE0=CROP(image=IMAGE,box=BOX0)
BOX1=LOC(image=IMAGE0,object='coat')
ANSWER0=COUNT(box=BOX1)
ANSWER1=EVAL(expr="'leather' if {ANSWER0} > 0 else 'wool'")
FINAL_RESULT=RESULT(var=ANSWER1)

Output:
{
  "error_type": "PLAN_STEP_DECOMPOSITION_ERROR, API_PARAMETER_ERROR",
  "error_layer": "TASK_PLANNING",
  "substep_ids": ["Step1", "Step3", "Step4"],
  "error_reason": "In Step 1, the LOC parameter omits the essential spatial anchor 'on the right', guaranteeing failure in multi-person scenes. Furthermore, the query 'What is the material...' is an open-ended attribute extraction. The plan arbitrarily restricts the universe of materials to 'leather' or 'wool' and relies on the mere existence of a coat bounding box (COUNT > 0) to output 'leather'. This is a complete logical hallucination.",
  "fix_strategy": "Step 1: Update the LOC parameter to 'woman on the right'. Step 3 & 4: Discard COUNT and EVAL. Forward the cropped image to the VQA tool and ask the open-ended question: 'What is the material of her coat?'."
}
""",
"""
Input:
1. User request:
Is the gap between the bed and the wall large enough for a nightstand?

2. Planned sub-steps:
Step1, Locate the gap between the bed and the wall.
Step2, Crop the gap region.
Step3, Ask VQA if it is large enough for a nightstand.

3. Generated code:
BOX0=LOC(image=IMAGE,object='gap between the bed and the wall')
IMAGE0=CROP(image=IMAGE,box=BOX0)
ANSWER0=VQA(image=IMAGE0,question='Is it large enough for a nightstand?')
FINAL_RESULT=RESULT(var=ANSWER0)

Output:
{
  "error_type": "PLAN_STEP_DECOMPOSITION_ERROR, NEGATIVE_SPACE_MATERIALIZATION",
  "error_layer": "TASK_PLANNING",
  "substep_ids": ["Step1", "Step2"],
  "error_reason": "In Step 1, the plan attempts to use the LOC tool to draw a bounding box around a 'gap'. A gap represents negative space—an absence of matter defined strictly by the proximity of surrounding objects. Object detection models (LOC) are trained exclusively on positive pixel clusters (physical matter). Instructing LOC to target empty space fundamentally breaches the tool's architecture, causing unpredictable crops that ruin Step 3.",
  "fix_strategy": "Never apply LOC tools to negative spaces, holes, or relative distances. To evaluate the properties of the space between objects, bypass LOC entirely and submit the uncropped scene directly to the VQA model."
}
""",
"""
Input:
1. User request:
Are the birds flying in a circular formation?

2. Planned sub-steps:
Step1, Locate the birds.
Step2, Count the number of boxes.
Step3, Determine yes or no using EVAL. If count > 5, answer yes.

3. Generated code:
BOX0=LOC(image=IMAGE,object='birds')
ANSWER0=COUNT(box=BOX0)
ANSWER1=EVAL(expr="'yes' if {ANSWER0} > 5 else 'no'")
FINAL_RESULT=RESULT(var=ANSWER1)

Output:
{
  "error_type": "PLAN_STEP_DECOMPOSITION_ERROR, GESTALT_PATTERN_DISRUPTION",
  "error_layer": "TASK_PLANNING",
  "substep_ids": ["Step2", "Step3"],
  "error_reason": "The user query explicitly asks to verify a macroscopic geometric pattern ('circular formation'). The plan mistakenly assumes that detecting a specific threshold quantity of birds (count > 5) mathematically guarantees a circular shape. This is a severe logical fallacy; 20 birds flying in a straight line or scattered randomly would trigger a false 'yes' output, completely ignoring the spatial layout intent of the question.",
  "fix_strategy": "Macroscopic patterns and formations cannot be validated by micro-level bounding box counts. Abandon COUNT and EVAL. Pass the full image directly to the VQA model, which possesses the gestalt visual comprehension required to recognize formations."
}
""",
"""
Input:
1. User request:
Is the dog looking at the tennis ball?

2. Planned sub-steps:
Step1, Locate the dog, and obtain bounding boxes.
Step2, Crop the region in front of the dog.
Step3, Locate the tennis ball in the cropped image.
Step4, Count the boxes and output yes if > 0.

3. Generated code:
BOX0=LOC(image=IMAGE,object='dog')
IMAGE0=CROP_FRONTOF(image=IMAGE,box=BOX0)
BOX1=LOC(image=IMAGE0,object='tennis ball')
ANSWER0=COUNT(box=BOX1)
ANSWER1=EVAL(expr="'yes' if {ANSWER0} > 0 else 'no'")
FINAL_RESULT=RESULT(var=ANSWER1)

Output:
{
  "error_type": "PLAN_STEP_DECOMPOSITION_ERROR, GAZE_VECTOR_SEVERANCE",
  "error_layer": "TASK_PLANNING",
  "substep_ids": ["Step2", "Step3", "Step4"],
  "error_reason": "Determining 'looking at' requires tracing an invisible 3D line-of-sight vector from the subject's eyes to the target. In Step 2, rigidly applying a 2D CROP_FRONTOF operation completely severs this spatial vector. The ball might be far away or at an angle, meaning the crop will miss it. Furthermore, simply proving a tennis ball exists somewhere in front of the dog (COUNT > 0) does not prove the dog is actively focusing its gaze on it.",
  "fix_strategy": "Do not use geometric cropping to evaluate line-of-sight, pointing, or gaze direction. Pass the entire unaltered image to the VQA model, explicitly asking it to evaluate if the dog's gaze is directed at the tennis ball."
}
""",
"""
Input:
1. User request:
Can you see the reflection of the trees in the lake?

2. Planned sub-steps:
Step1, Locate the lake, and crop it.
Step2, Locate the reflection of the trees in the cropped image.
Step3, Count the boxes.
Step4, Use EVAL to return yes if count is greater than 0.

3. Generated code:
BOX0=LOC(image=IMAGE,object='lake')
IMAGE0=CROP(image=IMAGE,box=BOX0)
BOX1=LOC(image=IMAGE0,object='reflection of the trees')
ANSWER0=COUNT(box=BOX1)
ANSWER1=EVAL(expr="'yes' if {ANSWER0} > 0 else 'no'")
FINAL_RESULT=RESULT(var=ANSWER1)

Output:
{
  "error_type": "PLAN_STEP_DECOMPOSITION_ERROR, OPTICAL_REALITY_CONFLATION",
  "error_layer": "TASK_PLANNING",
  "substep_ids": ["Step2", "Step3"],
  "error_reason": "In Step 2, the plan treats an optical illusion ('reflection') as a physical object to be detected by LOC. Object detectors are trained on solid entities possessing standard edge and textural features, not on the warped, rippled, inverted specular highlights found in water. Relying on COUNT > 0 from a tool not designed for optical phenomena will result in severe false negatives.",
  "fix_strategy": "Optical phenomena (reflections, refractions, translucency) require holistic reasoning. Remove the LOC 'reflection' and COUNT sequence. Pass the cropped lake image (or the full image) directly to the VQA tool and ask 'Can you see the reflection of the trees?'."
}
""",
"""
Input:

User request:
Are there more dogs than cats in the image?

Planned sub-steps:
Step1, Locate the dogs in the image and obtain bounding boxes.
Step2, Locate the cats in the image and obtain bounding boxes.
Step3, Compare the amount of dogs and cats directly using a Python expression. Output 'yes' if dogs are more than cats, else 'no'.
Step4, Visualize results.

Generated code:
BOX0=LOC(image=IMAGE,object='dog')
BOX1=LOC(image=IMAGE,object='cat')
ANSWER0=EVAL(expr="'yes' if {BOX0} > {BOX1} else 'no'")
FINAL_RESULT=RESULT(var=ANSWER0)

Output:
{
"error_type": "CODE_GENERATION_SPEC_ERROR, TYPE_MISMATCH_IN_EVAL",
"error_layer": "CODE_GENERATION",
"substep_ids": ["Step3"],
"error_reason": "Data type mismatch and API specification violation in EVAL: The user requested a numerical comparison between two object categories. While Steps 1 and 2 correctly localize the objects into variables BOX0 and BOX1, the generated code in Step 3 attempts to perform a mathematical greater-than (>) operation directly on these box variables. According to the API specification, BOX0 and BOX1 represent spatial bounding box arrays (complex lists of coordinate dictionaries), not scalar integers. Attempting to evaluate {BOX0} > {BOX1} is mathematically invalid and will result in a fatal TypeError during Python execution.",
"fix_strategy": "Adhere strictly to the tool chain specifications and variable types. Before any mathematical comparison can occur, the spatial array variables must be explicitly converted to scalar integers. Inject the COUNT tool (e.g., ANSWER0=COUNT(box=BOX0) and ANSWER1=COUNT(box=BOX1)) prior to the EVAL step, and then evaluate the resulting integer variables ({ANSWER0} > {ANSWER1})."
}
""",
"""
Input:

User request:
What are the exact bounding box coordinates of the red car parked on the street?

Planned sub-steps:
Step1, Ask the VQA tool to find the red car and output its exact coordinates.
Step2, Visualize results.

Generated code:
ANSWER0=VQA(image=IMAGE,question='What are the exact [x1, y1, x2, y2] coordinates of the red car parked on the street?')
FINAL_RESULT=RESULT(var=ANSWER0)

Output:
{
"error_type": "TOOL_SELECTION_ERROR",
"error_layer": "EXECUTION",
"substep_ids": ["Step1"],
"error_reason": "Architectural output mismatch: The user explicitly requested exact spatial bounding box coordinates ([x1, y1, x2, y2]). The plan mistakenly selects the VQA tool for this highly specialized geometric task. VQA models are architected as multimodal text generators designed to produce natural language semantic descriptions. They completely lack the continuous spatial coordinate regression capabilities required to output precise pixel boundaries. Forcing a VQA model to output coordinates will cause it to severely hallucinate a set of arbitrary numbers that do not align with the actual image space.",
"fix_strategy": "Select the appropriate tool designed specifically for spatial localization. Replace the VQA tool with the LOC tool (e.g., BOX0=LOC(image=IMAGE, object='red car')), which is mathematically trained to regress visual features into precise bounding box coordinate arrays."
}"""]

