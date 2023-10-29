from email.mime.text import MIMEText
import smtplib

def send_mail(context):

    # 确定发送方、邮箱授权码和接收方，邮件主题和内容
    my_from = '2653992054@qq.com'  # 发送方
    password = 'djnoxzzamnnhebgf'  # 授权码
    to = ['1183140624@qq.com']  # 接收方，可以多个接收方，类似于群发

    subject = 'Modularity Echo State Network'  # 主题
    # text = 'OK!'  # 正文
    text = context

    # 邮件内容设置
    # MIMEApplication用于发送各种文件，比如压缩，word，excel，pdf等
    '''
    用于发送图片的代码
    imageFile = r"1.jpg"
    imageApart = MIMEImage(open(imageFile, 'rb').read())
    imageApart.add_header('Content-Disposition', 'attachment', filename=imageFile)
    massage = MIMEMultipart()
    massage.attach(imageApart)
    构造邮件
    massage.attach(MIMEText(text,'html'))
    '''

    massage = MIMEText(text)  # 邮件对象
    massage['Subject'] = subject  # 给邮件添加主题
    # massage['From'] = my_from  # 谁发送的
    massage['From'] = "YLY"  # 谁发送的
    massage['To'] = ";".join(to)  # 发给你想发送的对象

    # 发送邮件
    s = smtplib.SMTP_SSL('smtp.qq.com', 465)
    # smtp.qq.com是qq邮箱的服务器地址（SMTP地址），465是他的端口号
    s.login(my_from, password)
    # 登录
    s.sendmail(my_from, to, massage.as_string())
    # 发送方地址，接收方地址，邮件内容
    massage.as_string() # 会将邮件原封不动的发送
    print("Send mail OK!\nFinish!")

# if __name__ == "__main__":
#     c = "HK!"
#     send_warning(c)