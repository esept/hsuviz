import plotly.graph_objects as go
import numpy as np
import torch
class hsuviz():
    
    def __init__(self,model,input_shape):
        '''
        定义类
        define the class
        '''
        self.__model = model
        self.__input = self.create_random_input(input_shape)
        self.shapes = self.__start()
        self.types = self.__get_module()
        self.__model_layers = self.get_nodes()
    
    def create_random_input(self,ie,aug = 30):
        '''
        生成一个随机的输入
        generate a random input 
        '''
        matrix = np.random.rand(*ie) * aug
        mm = torch.from_numpy(matrix)
        mm = mm.float()
        return mm

    def __start(self):
        '''
        得到每一层的输出大小形状
        get every layers' output size 
        '''
        this = []
        x = self.__input
        layers = self.__model.get_layers()
        for i in range(len(layers)):
            ll = layers[i]
            x = ll(x)
            this.append(x.shape)
        return this
            
    def __get_module(self):
        '''
        得到每一层的输出的类名称
        get every class of layers
        '''
        this = []
        for name,module in self.__model.named_modules():
            # print(f"{type(module)}")
            this.append(module)
        this.remove(this[0])
        return this
    
    def get_nodes(self):
        '''
        
        '''
        this= [l[0] for l in self.shapes]
        return this 
        # print(model_layers)
        
    # def have_node(self):
        
        
        
    
    def draw(self,show="False",save="True",name="exemple"):
        this_x = []
        this_y = []
        for i,layer in enumerate(self.__model_layers):
            layer_x = [i] * layer
            layer_y = list(range(layer))
            this_x.extend(layer_x)
            this_y.extend(layer_y)
            # print(layer_x,layer_y)
        node_trace = go.Scatter(
            x = this_x , y = this_y,
            mode = 'markers',
            marker = dict(size=10)
        )
        edge_x = []
        edge_y = []
        for i in range(len(self.__model_layers) - 1):
            for j in range(self.__model_layers[i]):
                for k in range(self.__model_layers[i + 1]):
                    edge_x.extend([i,i+1,None])
                    edge_y.extend([j,k,None])
        edge_trace = go.Scatter(x = edge_x,y = edge_y,
                                mode = "lines",line = dict(width = 1))
        fig = go.Figure(data=[edge_trace, node_trace])
        fig.update_layout(showlegend=False)
        if show :
            fig.show()
        if save :
            fig.write_image(name + ".jpg", width=1920, height=1080)
