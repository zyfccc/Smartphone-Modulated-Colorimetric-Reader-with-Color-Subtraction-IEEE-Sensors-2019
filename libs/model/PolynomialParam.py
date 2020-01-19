from libs.model.Num3 import Num3


class PolynomialParam:
    def __init__(self, W=[], b=0, scale=1, origin=None, json=None):
        if json is not None:
            self.fromJson(json)
        else:
            self.W = W
            self.b = b
            self.scale = scale
            self.origin = origin

    def toJson(self):
        return {
            'W': self.W,
            'b': self.b,
            'scale': self.scale,
            'origin': self.origin.toJson() if self.origin is not None else None,
        }

    def fromJson(self, json):
        self.W = json['W']
        self.b = json['b']
        self.scale = json['scale']
        origin = json.get('origin')
        self.origin = Num3(
            json=origin) if origin is not None else None
