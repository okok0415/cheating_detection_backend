# Generated by Django 3.2.7 on 2021-10-17 04:21

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('users', '0005_user_image'),
    ]

    operations = [
        migrations.AlterField(
            model_name='user',
            name='image',
            field=models.ImageField(blank=True, default='example.jpg', upload_to='photo/%Y/%m/%d'),
        ),
    ]