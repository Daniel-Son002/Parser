from time import time
import tensorflow as tf
from simple_elmo import ElmoModel
import time

elmo_model = ElmoModel()
elmo_model.load(r'C:\Users\liang\simple_elmo_training\bilm\209')

start = time.time()
print(elmo_model.get_elmo_vectors(["Hello, this is me."], layers="average"))
end = time.time()

processing_time = end - start
print(processing_time)