from collections import defaultdict

class Solution:

    # 1. Algorytm wyszukiwania palindromów
    def finding_palindromes(self, s: str) -> str:

        n, ans = len(s), []

        def addPalindrome(left: int, right: int) -> int:
            temp = []
            while left >= 0 and right < n and s[left] == s[right]:
                temp.append(s[left:right+1])
                left -= 1
                right += 1
            return temp
        
        for i in range(n):
            even = addPalindrome(i, i + 1)
            odd = addPalindrome(i, i)
            ans += even + odd
            
        return ans
    

    # 2. Algorytm wyszukiwania anagramów poprzez sortowanie liter z użyciem słownika
    def finding_anagrams_sort(self, words):
        
        anagram_dict = defaultdict(list)

        for word in words:
            anagram_dict[''.join(sorted(word))].append(word)
            
        return list(anagram_dict.values())

    # 3. Algorytm wyszukiwania anagramów poprzez zliczanie liter z użyciem słownika

    def finding_anagrams_dict(self, words: str) -> str:

        anagram_dict = {}

        for word in words:
            letter_count = [0] * 26
            for char in word:
                letter_count[ord(char) - ord('a')] += 1

            letter_count_tuple = tuple(letter_count)

            if letter_count_tuple in anagram_dict:
                anagram_dict[letter_count_tuple].append(word)
            else:
                anagram_dict[letter_count_tuple] = [word]

        return list(anagram_dict.values())
    
sol = Solution()

# 1.
s = "ccbbdddbddb"
result = sol.finding_palindromes(s)
print(" Wyszukane palindromy to:")
print(result)
print()

# 2.
words = ['eat', 'tea', 'tan', 'ate', 'nat', 'bat']
result = sol.finding_anagrams_sort(words)
print(" Znalezione anagramy to:")
print(result)
print()

# 3.
words = ['eat', 'tea', 'tan', 'ate', 'nat', 'bat']
result = sol.finding_anagrams_dict(words)
print(" Znalezione anagramy to:")
print(result)
print()