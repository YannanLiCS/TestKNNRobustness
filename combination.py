class combinationIterator:
    def __init__(self, list, cnt):
        self.list = list
        self.next_comb = [i for i in range(cnt)] #first combination
        self.empty_comb = True

    def hasNext(self):
        if len(self.next_comb) == 0:
            return self.empty_comb
        else:
            return self.next_comb[0] + len(self.next_comb) <= len(self.list)
        
    def getNext(self):
        if len(self.next_comb) == 0:
            self.empty_comb = False
            return []
        comb = [self.list[i] for i in self.next_comb]
        length = len(self.next_comb)
        index = 0
        for i in range(length):
            if self.next_comb[-1 - i] != len(self.list) - 1 - i:
                index = length - 1 - i
                break
        self.next_comb[index] += 1
        for i in range(index + 1, length, 1):
            self.next_comb[i] = self.next_comb[i-1] + 1
        return comb
        
'''
# test
cI = combinationIterator(['a', 'b', 'c', 'd', 'e'], 0)
while cI.hasNext():
    print(cI.getNext())
'''

