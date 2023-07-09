import time
from tqdm.auto import tqdm

for iterations in tqdm(range(10), desc=" iterations", position=0):
    for outerloop in tqdm(range(5), desc=" outer loops", position=1, leave=False):
        for labels in tqdm(range(10), desc=" labels", position=2, leave=False):
            time.sleep(0.05)
        for innerloop in tqdm(range(5), desc=" inner loops", position=2, leave=False):
            time.sleep(0.05)
print("done!")
