import yagmail
import getpass
import cv2 
def SendMail(img):
    try:
        #initializing the server connection
        #pswd = getpass.getpass('Password: ')
        # pswd = 
        # contents = [yagmail.inline("/ali.161.8.jpg")]
        cv2.imwrite('output.jpg',img)
        yag = yagmail.SMTP(user='ushakeel6060@gmail.com', password='passwordhacked123456')
        #sending the email
        yag.send(to='ushakeel808@gmail.com', subject='Security Alert'
        , 
        contents=[
        "Hello Mike! Here is a picture I took last week:",
        "output.jpg"
        
        
        ]

        # ,contents= "asjlkdjasklj"

        )
        print("Email sent successfully")
    except Exception as e:
        print(e)