import heapq

class Solution:
    # Sainte-Lague
    def sainte_lague(self, votes, seats):
        allocated_seats = [0] * len(votes)
        heap = []
        heapq.heapify(heap)

        for i, vote in enumerate(votes):
            heapq.heappush(heap, (-vote/(1.4), i))

        for _ in range(seats):
            _, party = heapq.heappop(heap)
            allocated_seats[party] += 1
            val = votes[party]/(2*allocated_seats[party] + 1)
            heapq.heappush(heap, (-val, party))

        return allocated_seats

    # D'Hondt
    def d_hondt(self, votes, seats):
        
        allocated_seats = [0] * len(votes)
        heap = []
        heapq.heapify(heap)

        for i, vote in enumerate(votes):
            heapq.heappush(heap, (-vote, i))

        for _ in range(seats):
            _, party = heapq.heappop(heap)
            allocated_seats[party] += 1
            val = votes[party]/(allocated_seats[party] + 1)
            heapq.heappush(heap, (-val, party))

        return allocated_seats

    # Hare-Niemeyer
    def hare_niemeyer(self, votes, seats):
        total_votes = sum(votes)
        hare_quota = total_votes / seats
        allocated_seats = [int(v // hare_quota) for v in votes]
        remaining_seats = seats - sum(allocated_seats)
        remainders = [v % hare_quota for v in votes]

        while remaining_seats > 0:
            largest_index = remainders.index(max(remainders))
            allocated_seats[largest_index] += 1
            remaining_seats -= 1
            remainders[largest_index] = 0

        return allocated_seats

sol = Solution()

# 1. 
votes = [7_640_854, 6_629_402, 3_110_670, 1_859_018, 1_547_364]
seats = 460
sainte_lague_results = sol.sainte_lague(votes, seats)
print(sainte_lague_results)
print()


# 2.
votes = [7_640_854, 6_629_402, 3_110_670, 1_859_018, 1_547_364]
seats = 460
d_hondt_results = sol.d_hondt(votes, seats)
print(d_hondt_results)
print()


# 3.
votes = [7_640_854, 6_629_402, 3_110_670, 1_859_018, 1_547_364]
seats = 460
hare_niemeyer_results = sol.hare_niemeyer(votes, seats)
print(hare_niemeyer_results) 
print()

