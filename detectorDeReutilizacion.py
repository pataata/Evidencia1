from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

class detectorDeReutilizacion():
  
  def __init__(self):
    self.model = Sequential()
    self.model.add(Dense(12, input_shape=(3,), activation='relu'))
    self.model.add(Dense(8, activation='relu'))
    self.model.add(Dense(1, activation='sigmoid'))
    self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

  def entrenar(self, x, y, epochs = 1, batch_size = 1):
    self.model.fit(x, y, epochs=epochs, batch_size=batch_size)

  def evaular(self, x, y):
    _, accuracy = self.model.evaluate(x, y)
    print('Accuracy: %.2f' % (accuracy*100))
  
  def predict(self, x):
    return self.model.predict(x)
  
  def save_model(self, name = "detector_texto_reutilizado_e1.h5b"):
    self.model.save('detector_texto_reutilizado_e1.h5')
    print(f"Modelo exportado cómo {name}")