import openai
import requests,cv2
import os
from dotenv import load_dotenv
load_dotenv()


API_URL = "https://api-inference.huggingface.co/models/facebook/detr-resnet-50"
headers = {"Authorization": f"Bearer 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"}

def query(filename):
    with open(filename, "rb") as f:
        data = f.read()
    response = requests.post(API_URL, headers=headers, data=data)
    # print('Data:',response.json())
    return response.json()

dir = "C:\\Users\\ABHISHEK CHANDRA\\Desktop\\Vs code\\storytelling\\images\\"
image_files = os.listdir(dir)
# print('Images found are listed below.')
# for img in image_files:
#     print(img)
story = []
for img in image_files:
    link= dir + img
    print('Image:', link)
    output = query(link)
    object_count = {}
    for data in output:
        if object_count.get(data['label']) == None:
            object_count[data['label']] = 1
        else:
            object_count[data['label']] += 1
    if len(object_count):
        print('Object with count that are found in the image are given below.')
        # print('----------------------------------------------------------------------------------------------------------------------------------')
        for obj in object_count:
            print('Obejct:',obj,'   Count:',object_count[obj])
    # print(output)
    image = cv2.imread(link)
    for data in output:
        xmin,ymin,xmax,ymax = data['box']['xmin'], data['box']['ymin'], data['box']['xmax'], data['box']['ymax']
        sp = (xmin,ymin)
        ep = (xmax, ymax)

        color = (255, 0, 0) 
    
        # Line thickness of 4 px 
        thickness = 4
        
        # Using cv2.rectangle() method 
        # Draw a rectangle with blue line borders of thickness of 2 px 
        image = cv2.rectangle(image, sp, ep, color, thickness)

    # Displaying the image  
    image = cv2.resize(image, (600, 600))
    cv2.imshow('image', image)
    cv2.waitKey(5000)
    cv2.destroyAllWindows()
    
    # Step 4: Integrate OpenAI with the code
    openai.api_key = os.environ.get('OPENAI_API_KEY')

    scene_prompt = 'Create a one sentence story from the given link: ' + link

    response = openai.Completion.create(
      engine="gpt-3.5-turbo-instruct",
      prompt=scene_prompt,
      max_tokens=100
    )

    scene_story = response.choices[0]['text']
    print('\n')
    print('Scene Story:', scene_story)
    story.append(scene_story)
    print('----------------------------------------------------------------------------------------------------------------------------------')

print('Combined Story')

scene_prompt = 'Combine the stories in the given list and make a single story of 100 words.: ' + scene_story

response = openai.Completion.create(
    engine="gpt-3.5-turbo-instruct",
    prompt=scene_prompt,
    max_tokens=200
)

combined_story = response.choices[0]['text']
print(combined_story)