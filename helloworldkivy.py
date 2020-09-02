from kivy.app import App
from kivy.uix.button import Button
from kivy.config import Config
Config.set('graphics', 'width', '1400')
Config.set('graphics', 'height', '900')

class TestApp(App):
    def build(self):
        return Button(text='Hello World')

TestApp().run()