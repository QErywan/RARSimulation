import torch
import os

try:
    from .csh.csvec import CSVec
except:
    os.system("git clone https://github.com/nikitaivkin/csh {}".format(os.path.join(os.path.abspath(os.path.dirname(__file__)), "./csh")))    
    from .csh.csvec import CSVec

##############################################################################
##############################################################################

class CSVec_Extended(CSVec):
    
    def to_1d_tensor(self):
        return self.table.view(-1)

    def from_1d_tensor(self, tbl):
        self.table = torch.reshape(tbl, self.table.shape)

##############################################################################
##############################################################################

class CountSketchSender:

    
    def __init__(self, device='cpu'):

        self.device = device
                
            
    def compress(self, data):
        
        sketch = CSVec_Extended(d=data["d"], c=data["c"], r=data["r"], device=self.device)
        sketch.accumulateVec(data['vec'])
                                                
        return {'vec': sketch.to_1d_tensor(), 'data': data}

##############################################################################
##############################################################################
       
class CountSketchReceiver:

    
    def __init__(self, device='cpu'):

        self.device = device
                
            
    def decompress(self, data):
                        
        sketch = CSVec_Extended(d=data["data"]["d"], c=data["data"]["c"], r=data["data"]["r"], device=self.device)
        sketch.from_1d_tensor(data["vec"])
        
        return sketch.unSketch(k=data["data"]["d"])
        
##############################################################################
##############################################################################

class SimpleCountSketchSender:

    
    def __init__(self, device='cpu'):

        self.device = device
        self.prng = torch.Generator(device=device)
        self.large_prime = 2**61-1


    def compress(self, data):
        
        self.prng.manual_seed(data["seed"])

        d = data["vec"].numel()
        rows = data["r"]
        cols = data["c"]
            
        if data["fast_prng"]:
                        
            tokens = torch.arange(d, device=self.device)
            hashes = torch.randint(0, self.large_prime, (rows, 6), device=self.device, generator=self.prng)
            
            # computing sign hashes (4 wise independence)
            
            h1 = hashes[:,2:3]
            h2 = hashes[:,3:4]
            h3 = hashes[:,4:5]
            h4 = hashes[:,5:6]
            
            randSigns = (((h1 * tokens + h2) * tokens + h3) * tokens + h4)
            randSigns = ((randSigns % self.large_prime % 2) * 2 - 1).float()
       
            # computing index hashes (2-wise independence)
            
            h1 = hashes[:,0:1]
            h2 = hashes[:,1:2]
            
            hashedIndices = ((h1 * tokens) + h2) % self.large_prime % cols
    
        else:
            
            randSigns = 2 * torch.randint(low=0, high=2, size=(rows,d), generator=self.prng, device=self.device) - 1 
            hashedIndices = torch.randint(low=0, high=cols, size=(rows,d), generator=self.prng, device=self.device)  

        ### sketch
        sketch = torch.zeros(size=(rows, cols)).to(self.device)        
        for row in range(rows):
            sketch[row] += torch.bincount(hashedIndices[row], weights=randSigns[row] * data["vec"], minlength=cols)
                                                                
        return {'sketch': sketch, 'data': data, 'dim': d}

##############################################################################
##############################################################################
       
class SimpleCountSketchReceiver:

    
    def __init__(self, device='cpu'):

        self.device = device
        self.prng = torch.Generator(device=device)
        self.large_prime = 2**61-1
                
            
    def decompress(self, data):
                        
        self.prng.manual_seed(data["data"]["seed"])

        d = data["dim"]
        rows = data["data"]["r"]
        cols = data["data"]["c"]
            
        if data["data"]["fast_prng"]:
                        
            tokens = torch.arange(d, device=self.device)
            hashes = torch.randint(0, self.large_prime, (rows, 6), device=self.device, generator=self.prng)
            
            # computing sign hashes (4 wise independence)
            
            h1 = hashes[:,2:3]
            h2 = hashes[:,3:4]
            h3 = hashes[:,4:5]
            h4 = hashes[:,5:6]
            
            randSigns = (((h1 * tokens + h2) * tokens + h3) * tokens + h4)
            randSigns = ((randSigns % self.large_prime % 2) * 2 - 1).float()
       
            # computing index hashes (2-wise independence)
            
            h1 = hashes[:,0:1]
            h2 = hashes[:,1:2]
            
            hashedIndices = ((h1 * tokens) + h2) % self.large_prime % cols
    
        else:
            
            randSigns = 2 * torch.randint(low=0, high=2, size=(rows,d), generator=self.prng, device=self.device) - 1 
            hashedIndices = torch.randint(low=0, high=cols, size=(rows,d), generator=self.prng, device=self.device)  
        
        ### unsketch
        uvec = torch.zeros(size=(rows, d)).to(self.device)
        for row in range(rows):
            uvec[row] = torch.take(data["sketch"][row], hashedIndices[row])*randSigns[row]
        
        return uvec.median(dim=0)[0]
        
##############################################################################
##############################################################################        
