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

Link to the project:
https://colab.research.google.com/drive/1HRGiMSLQ72LJmVEEZWZ8SNbEgQG9dtnP?usp=sharing

Link to the testing image data: 
http://cdn.shibe.online/shibes/e4a90fa2c4f2493172d9e85a7e93c2a47eede3a5.jpg
http://cdn.shibe.online/shibes/069011c08d599e45433af3d4563d66a399736d7c.jpg
http://cdn.shibe.online/shibes/fcac2c67aeb459dcfc4eaed2fb4f9a916aaffba2.jpg
http://cdn.shibe.online/shibes/53770517d03b2a619aed39d6eb67bf4a92880788.jpg
http://cdn.shibe.online/shibes/3407be2984602da9c1b3287a5c96277c625c143a.jpg
http://cdn.shibe.online/shibes/a469aee1c44e6eeb13226b2931e4daaa7eb4dbfc.jpg
http://cdn.shibe.online/shibes/e4dfe4ef53d038061d72c0dd61e369e61da3ccfb.jpg
http://cdn.shibe.online/shibes/e07a9becbbf6a558ec6fa6c98c0fb294c73c2c92.jpg
http://cdn.shibe.online/shibes/73a358336feb703c05f7924aa10bcf5aa2c0353b.jpg
http://cdn.shibe.online/shibes/833ec78214cb99aaa283e28d66bfb828a9bc3642.jpg
http://cdn.shibe.online/shibes/059873a73e606f1dc3d2cf19b1e65afbba61ba27.jpg
 http://cdn.shibe.online/shibes/ff3fa37f208bc8b91aeacfb83c848c2cccbbb8ec.jpg
http://cdn.shibe.online/shibes/87f19e130d16f5c9f9ae01fa34569bdd4994e531.jpg

Link to Google spreadsheet with results:
https://docs.google.com/spreadsheets/d/17hwk-V8JTKVEBFhAea9TUNgkXCQezsRcDutZSLjBzSI/edit?usp=sharing

