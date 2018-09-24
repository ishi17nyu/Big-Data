select HDate, percentageChange from Eqix where percentageChange = (select max(percentageChange) from Eqix);
select HDate, percentageChange from Flxs where percentageChange = (select max(percentageChange) from Flxs);
select HDate, percentageChange from Ifon where percentageChange = (select max(percentageChange) from Ifon);
