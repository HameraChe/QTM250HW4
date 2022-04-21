# QTM250HW4

The following demonstrates how to explore whether or not some variables can affect the accuracy of the machine learning model from Google Cloud Vision API. 

In this case, we consider two variables:
1)	Image quality
2)	Surrounding circumstances of dogs in the images

Tools needed:
Shibe.Online API (https://shibe.online/)
Google Cloud Vision API
Google Collaboration
Google Spreadsheet
Google Cloud Storage (through Google Cloud Console) 

This project is implemented in Google Collaboration in Python language. 

The testing data are retrieved from an additional API from Shibe that can generate random images. The image files are later stored in the Google Cloud Storage bucket. 

After importing Google Cloud Vision API to the workspace (Google Collaboration), the testing data can be evaluated by the model with the following code.

for x in df['gs']:
  IMAGE= x
  service = build('vision', 'v1',developerKey=APIKEY)
  request = service.images().annotate(body={
        'requests': [{
                'image': {
                    'source': {
                        'gcs_image_uri': IMAGE
                    }
                },
                'features': [{
                    'type': 'OBJECT_LOCALIZATION',
                    'maxResults': 3,
                }]
            }],
        })
  responses = request.execute()
  annotation = responses['responses'][0]['localizedObjectAnnotations'][0]['name']
  score = responses['responses'][0]['localizedObjectAnnotations'][0]['score']
  print(annotation)
  print(score)

The code above returns the identification of the image (should be dog) and the confidence score that it is a dog. 

The results are later exported to the Google spreadsheet for the statistical and visual analysis with internal functions of spreadsheet. 
