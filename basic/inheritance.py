# -*- coding: utf-8 -*-

"""
@Author: xiezizhe
@Date: 6/2/2020 下午11:39
"""


class Contact:
    all_contacts = []

    def __init__(self, name, email, **kwargs):
        super().__init__(**kwargs)
        self.name = name
        self.email = email
        Contact.all_contacts.append(self)


class MailSender:
    def __init__(self, no, **kwargs):
        super().__init__(**kwargs)
        self.no = no

    def send_mail(self, message):
        print("no.{} sending '{}'".format(self.no, message))


class Supplier(Contact, MailSender):
    def __init__(self, **kwargs):
        """
        three parameters are required
        :param kwargs: name, email, no
        """
        super().__init__(**kwargs)

    def order(self, order):
        print("{} wants to order {}".format(self.name, order))


if __name__ == "__main__":
    o = Supplier(name='ming', email='ming@126.com', no='1')
    o.order("apple")
    o.send_mail("hello, world")
