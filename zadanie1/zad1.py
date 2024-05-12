class Solution:

    # 1. Algorytm wyznaczania lat przestepnych Grzegorza XIII
    def leap_grzegorz(self, year: int) -> str:

        if (year % 4 == 0 and year % 100 != 0) or year % 400 == 0:
            return f"Rok {year} jest przestępny."
        
        return f"Rok {year} nie jest przestępny."
        

    # 2. Algorytm Gaussa wyznaczania daty Wielkanocy
    def gauss_easter(self, year: int) -> str:

        a = year % 19
        b = year % 4
        c = year % 7
        k = year//100
        p = (13 + 8*k)//25
        q = k//4
        M = (15 - p + k - q) % 30
        N = (4 + k - q) % 7
        d = (19*a + M) % 30
        e = (2*b + 4*c + 6*d + N) % 7
        
        day = 22 + d + e
        month = 3

        if d == 28 and e == 6 and (11*M + 11) % 30 < 19 and day == 25:
            day = 18

        if d == 29 and e == 6 and day == 26:
            day = 19

        if day > 31:
            month += 1
            day -= 31        


        return f"{day:02d}.{month:02d}.{year:04d}"
    

    # 3. Algorytm Meeusa-Jones’a-Butchera wyznaczania daty Wielkanocy
    def mjb_easter(self, year: int) -> str:

        a = year % 4
        b = year % 7
        c = year % 19
        d = (19*c + 15) % 30
        e = (2*a + 4*b - d + 34) % 7

        month = (d + e + 114)//31
        day = (d + e + 114) % 31 + 1

        return f"{day:02d}.{month:02d}.{year:04d}"
    

# Test algorytmów

sol = Solution()
years = [2020, 2021, 2022, 2023, 2024]

for year in years:
    is_leap = sol.leap_grzegorz(year)
    gauss_easter_date = sol.gauss_easter(year)
    mjb_easter_date = sol.mjb_easter(year)

    print(f"{is_leap}")
    print(f"Data Wielkanocy według Gaussa: {gauss_easter_date}")
    print(f"Data Wielkanocy według Meeusa-Jones’a-Butchera: {mjb_easter_date}")
    print()