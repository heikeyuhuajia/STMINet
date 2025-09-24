
class DataConfig:
    data_name = ""
    root_dir = ""
    label_transform = "norm"
    def get_data_config(self, data_name):
        self.data_name = data_name
        if data_name == 'LEVIR':
            self.root_dir = 'xx'  # Your Path
        elif data_name == 'SYSU-CD':
            self.root_dir = 'xx'
        elif data_name == 'DSIFN':
            self.root_dir = 'xx'
        elif data_name == 'GZCD':
            self.root_dir = 'xx'
        elif data_name == 'WHU':
            self.root_dir = 'xx'
        elif data_name == 'GT':
            self.root_dir = 'D:/data/CD/GT/'    
        elif data_name == 'quick_start':
            self.root_dir = './samples/'
        else:
            raise TypeError('%s has not defined' % data_name)
        return self


if __name__ == '__main__':
    data = DataConfig().get_data_config(data_name='LEVIR')
    print(data.data_name)
    print(data.root_dir)
    print(data.label_transform)

