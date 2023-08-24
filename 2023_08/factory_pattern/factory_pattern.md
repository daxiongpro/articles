# 设计模式——工厂模式

## 问题背景

现在有三款手机，小米，苹果，华为。需要编程查询每款手机的价格。你应该会这样编程：

```python
def get_price(type):
    if type == 'xiaomi':
        print('1999')
    elif type == 'huawei':
        print('5999')
    elif type == 'apple':
        print('8999')
    else:
        print('error type!')


price = get_price('xiaomi')
```

这样编程有个问题：当你想扩展手机品牌的时候，你就需要修改 get_price 函数的源代码。但你这样不利于代码的扩展。因此，可以将这三款手机拆开，建立三个手机类。这就是简单工厂模式。

## 简单工厂模式

分析上面的问题，可以得到以下代码：

```python
class MobileBase:
    def get_price():
        pass


class Xiaomi(MobileBase):
    def get_price():
        print('1999')


class Huawei(MobileBase):
    def get_price():
        print('5999')


class Apple(MobileBase):
    def get_price():
        print('8999')


class MobileFactory:
    def __init__(self, type) -> None:
        self.type = type

    def get_mobile(self):
        if self.type == 'xiaomi':
            return Xiaomi()
        elif self.type == 'huawei':
            return Huawei()
        elif self.type == 'apple':
            return Apple()
        else:
            print('error')


mobile = MobileFactory('xiaomi')
price = mobile.get_price()
```

有人会说：在 MobielFactory 类中，代码还是杂糅在一起，也没有实现解耦，样不是多次一举吗？因此需要普通工厂模式出场了。

## 普通工厂模式

普通工厂模式，就是把工厂类也拆开，拆成小米，华为，苹果三个工厂。

```python
class MobileBase:
    def get_price():
        pass


class Xiaomi(MobileBase):
    def get_price():
        print('1999')


class Huawei(MobileBase):
    def get_price():
        print('5999')


class Apple(MobileBase):
    def get_price():
        print('8999')


class XiaomiFactory:
    def get_mobile(self):
        return Xiaomi()


class HuaweiFactory:
    def get_mobile(self):
        return Huawei()


class AppleFactory:
    def get_mobile(self):
        return Apple()

mobile = XiaomiFactory()
price = mobile.get_price()
```

有人会说：我直接调用手机类不行吗？弄个工厂多此一举干什么？但是我们要知道，每个工厂不止生产一种产品，可以生产与此工厂相关的很多产品。比如：小米工厂可以生产小米手机、小米电脑、小米智能马桶……华为工厂可以生产华为手机、华为平板、华为麒麟芯片……苹果工厂可以生产苹果手机、苹果手表、苹果耳机……这样就可以模块化了。

```python
class XiaomiFactory:
    def get_mobile(self):
        return XiaomiMobile()
    def get_pc(self):
        return XiaomiPC()


class HuaweiFactory:
    def get_mobile(self):
        return HuaweiMobile()
    def get_pad(self):
        return HuaweiPad()


class AppleFactory:
    def get_mobile(self):
        return Apple()
    def get_airpods(self):
        return AirPods()


mobile = XiaomiFactory().get_mobile()
price = mobile.get_price()
```

## 题外话：抽象工厂模式

上面的手机类都是继承于 MobileBase 这个基类。这样做的好处是使得多人开发的时候代码编写统一，比如，统一获取价格的函数名称叫 get_price()。在 Java 中，最好是使用接口，Python 中没有接口，就使用这样一个抽象类。

抽象工厂与之类似，就是把工厂类的公共部分抽象出来。比如每个工厂需要统计员工的数量，可以这样：

```python
class FactoryBase:
    def count_employee():
        pass
  
class XiaomiFactory(FactoryBase):
    def count_employee():
        print('1999 人')
    def get_mobile(self):
        return XiaomiMobile()
    def get_pc(self):
        return XiaomiPC()


class HuaweiFactory(FactoryBase):
    def count_employee():
        print('5999 人')
    def get_mobile(self):
        return HuaweiMobile()
    def get_pad(self):
        return HuaweiPad()


class AppleFactory(FactoryBase):
    def count_employee():
        print('8999 人')
    def get_mobile(self):
        return Apple()
    def get_airpods(self):
        return AirPods()


xiaomi_employee = XiaomiFactory().count_employee()
```

## 参考文献

[bilibili](https://www.bilibili.com/video/BV1hM411V7Mk/?spm_id_from=333.788&vd_source=da7944bcc998e29818ec76ea9c6f1f47)

## 日期

2023/08/24：文章撰写日期
