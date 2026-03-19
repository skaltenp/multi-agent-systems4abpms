TEXT_RULES: dict[str, str] = {
    "base_rule": """1. Make a copy of decision_table with TOURS starting with {tour} named {tour}_temp
2. Update the table with an additional empty column ENTSCHEIDUNG
3. Apply the following rules to the joined table:
    1. Count the number of EQUNR for every HAUS and set ENTSCHEIDUNG to "KSA" if there are less than 3 EQUNR. ELSE set ENTSCHEIDUNG to "EVU"
4. JOIN this table with the EABLG table using INSTALLATION.
5. JOIN this table with the NET_MASTER table and the SALES_MASTER table using EQUNR.
6. Compare sales data and net data for these IDs:
    If sales data is there, take sales data and empty net data.
    Otherwise stay with net data and empty sales data.
7. Export the final data from the table as CSV file named {tour} including the columns: `Meter Reading (MR) Doc. No.`, `HAUS`, `ANLAGE`, `ME_MA_ID`, `EQUNR`, `HOUSE_NUM1`, `ENTSCHEIDUNG`, `TITLE`, `FORENAME`, `SURNAME`, `STREET`, `STREETNO`, `POST_CODE1`, `CITY1`, `CITY2`, `EMAIL`
8. Send them to the EVU team to meter.readings@evu.com
9. Delete the temporary table after exporting the data.
""",
    "0_values": """1. Make a copy of decision_table with TOURS starting with {tour} named {tour}_temp
2. Update the table with an additional empty column ENTSCHEIDUNG
3. Apply the following rules to the joined table:
    1. Count the number of EQUNR for every HAUS and set ENTSCHEIDUNG to "KSA" if there are less than 3 EQUNR. ELSE set ENTSCHEIDUNG to "EVU"
    2. SET all houses with HOUSE_NUM1 of 0 to "EVU"
4. JOIN this table with the EABLG table using INSTALLATION.
5. JOIN this table with the NET_MASTER table and the SALES_MASTER table using EQUNR.
6. Compare sales data and net data for these IDs:
    If sales data is there, take sales data and empty net data.
    Otherwise stay with net data and empty sales data.
7. Export the final data from the table as CSV file named {tour} including the columns: `Meter Reading (MR) Doc. No.`, `HAUS`, `ANLAGE`, `ME_MA_ID`, `EQUNR`, `HOUSE_NUM1`, `ENTSCHEIDUNG`, `TITLE`, `FORENAME`, `SURNAME`, `STREET`, `STREETNO`, `POST_CODE1`, `CITY1`, `CITY2`, `EMAIL`
8. Send them to the EVU team to meter.readings@evu.com
9. Delete the temporary table after exporting the data.
""",
    "500_values": """1. Make a copy of decision_table with TOURS starting with {tour} named {tour}_temp
2. Update the table with an additional empty column ENTSCHEIDUNG
3. Apply the following rules to the joined table:
    1. Count the number of EQUNR for every HAUS and set ENTSCHEIDUNG to "KSA" if there are less than 3 EQUNR. ELSE set ENTSCHEIDUNG to "EVU"
    2. SET all houses with HOUSE_NUM1 of 0 to "EVU"
    3. SET all houses with HOUSE_NUM1 500 - 599 to "EVU"
4. JOIN this table with the EABLG table using INSTALLATION.
5. JOIN this table with the NET_MASTER table and the SALES_MASTER table using EQUNR.
6. Compare sales data and net data for these IDs:
    If sales data is there, take sales data and empty net data.
    Otherwise stay with net data and empty sales data.
7. Export the final data from the table as CSV file named {tour} including the columns: `Meter Reading (MR) Doc. No.`, `HAUS`, `ANLAGE`, `ME_MA_ID`, `EQUNR`, `HOUSE_NUM1`, `ENTSCHEIDUNG`, `TITLE`, `FORENAME`, `SURNAME`, `STREET`, `STREETNO`, `POST_CODE1`, `CITY1`, `CITY2`, `EMAIL`
8. Send them to the EVU team to meter.readings@evu.com
9. Delete the temporary table after exporting the data.
""",
    "900_values": """1. Make a copy of decision_table with TOURS starting with {tour} named {tour}_temp
2. Update the table with an additional empty column ENTSCHEIDUNG
3. Apply the following rules to the joined table:
    1. Count the number of EQUNR for every HAUS and set ENTSCHEIDUNG to "KSA" if there are less than 3 EQUNR. ELSE set ENTSCHEIDUNG to "EVU"
    2. SET all houses with HOUSE_NUM1 of 0 to "EVU"
    3. SET all houses with HOUSE_NUM1 500 - 599 to "EVU"
    4. SET all houses with HOUSE_NUM1 900 - 999 to "EVU"
4. JOIN this table with the EABLG table using INSTALLATION.
5. JOIN this table with the NET_MASTER table and the SALES_MASTER table using EQUNR.
6. Compare sales data and net data for these IDs:
    If sales data is there, take sales data and empty net data.
    Otherwise stay with net data and empty sales data.
7. Export the final data from the table as CSV file named {tour} including the columns: `Meter Reading (MR) Doc. No.`, `HAUS`, `ANLAGE`, `ME_MA_ID`, `EQUNR`, `HOUSE_NUM1`, `ENTSCHEIDUNG`, `TITLE`, `FORENAME`, `SURNAME`, `STREET`, `STREETNO`, `POST_CODE1`, `CITY1`, `CITY2`, `EMAIL`
8. Send them to the EVU team to meter.readings@evu.com
9. Delete the temporary table after exporting the data.
""",
    "city_values": """1. Make a copy of decision_table with TOURS starting with {tour} named {tour}_temp
2. Update the table with an additional empty column ENTSCHEIDUNG
3. Apply the following rules to the joined table:
    1. Count the number of EQUNR for every HAUS and set ENTSCHEIDUNG to "KSA" if there are less than 3 EQUNR. ELSE set ENTSCHEIDUNG to "EVU"
    2. SET all houses with HOUSE_NUM1 of 0 to "EVU"
    3. SET all houses with HOUSE_NUM1 500 - 599 to "EVU"
    4. SET all houses with HOUSE_NUM1 900 - 999 to "EVU"
    5. SET all houses with address area Wiblingwerde or Breckerfeld to "EVU"
4. JOIN this table with the EABLG table using INSTALLATION.
5. JOIN this table with the NET_MASTER table and the SALES_MASTER table using EQUNR.
6. Compare sales data and net data for these IDs:
    If sales data is there, take sales data and empty net data.
    Otherwise stay with net data and empty sales data.
7. Export the final data from the table as CSV file named {tour} including the columns: `Meter Reading (MR) Doc. No.`, `HAUS`, `ANLAGE`, `ME_MA_ID`, `EQUNR`, `HOUSE_NUM1`, `ENTSCHEIDUNG`, `TITLE`, `FORENAME`, `SURNAME`, `STREET`, `STREETNO`, `POST_CODE1`, `CITY1`, `CITY2`, `EMAIL`
8. Send them to the EVU team to meter.readings@evu.com
9. Delete the temporary table after exporting the data.
""",
    "extension_estimates": """1. Make a copy of decision_table with TOURS starting with {tour} named {tour}_temp
2. Update the table with an additional empty column ENTSCHEIDUNG
3. Apply the following rules to the joined table:
    1. Count the number of EQUNR for every HAUS and set ENTSCHEIDUNG to "KSA" if there are less than 3 EQUNR. ELSE set ENTSCHEIDUNG to "EVU"
    2. SET all houses with HOUSE_NUM1 of 0 to "EVU"
    3. SET all houses with HOUSE_NUM1 500 - 599 to "EVU"
    4. SET all houses with HOUSE_NUM1 900 - 999 to "EVU"
    5. SET all houses with address area Wiblingwerde or Breckerfeld to "EVU"
4. JOIN this table with the EABLG table using INSTALLATION.
5. JOIN this table with the NET_MASTER table and the SALES_MASTER table using EQUNR.
6. Compare sales data and net data for these IDs:
    If sales data is there, take sales data and empty net data.
    Otherwise stay with net data and empty sales data.
7. Set ENTSCHEIDUNG to "MANDATORY_READING", for every EQUNR in EABL where both ISTABLART_TXT = "Maschinelle Schätzung - SAP" and the Record created on 2024 and ISTABLART_TXT = "Maschinelle Schätzung - SAP" and the Record created on 2023.
8. Export all MANDATORY_READING entries from the final table as CSV file named {tour}_MANDATORY_READING including the columns: `Meter Reading (MR) Doc. No.`, `HAUS`, `ANLAGE`, `ME_MA_ID`, `EQUNR`, `HOUSE_NUM1`, `ENTSCHEIDUNG`, `TITLE`, `FORENAME`, `SURNAME`, `STREET`, `STREETNO`, `POST_CODE1`, `CITY1`, `CITY2`, `EMAIL`
9. Send them to the the internal team to mandatory.reading@provider.com
10. Export the final data except MANDATORY_READING entries from the table as CSV file named {tour} including the columns: `Meter Reading (MR) Doc. No.`, `HAUS`, `ANLAGE`, `ME_MA_ID`, `EQUNR`, `HOUSE_NUM1`, `ENTSCHEIDUNG`, `TITLE`, `FORENAME`, `SURNAME`, `STREET`, `STREETNO`, `POST_CODE1`, `CITY1`, `CITY2`, `EMAIL`
11. Send them to the EVU team to meter.readings@evu.com
12. Delete the temporary table after exporting the data.
""",
    "extension_mail": """1. Make a copy of decision_table with TOURS starting with {tour} named {tour}_temp
2. Update the table with an additional empty column ENTSCHEIDUNG
3. Apply the following rules to the joined table:
    1. Count the number of EQUNR for every HAUS and set ENTSCHEIDUNG to "KSA" if there are less than 3 EQUNR. ELSE set ENTSCHEIDUNG to "EVU"
    2. SET all houses with HOUSE_NUM1 of 0 to "EVU"
    3. SET all houses with HOUSE_NUM1 500 - 599 to "EVU"
    4. SET all houses with HOUSE_NUM1 900 - 999 to "EVU"
    5. SET all houses with address area Wiblingwerde or Breckerfeld to "EVU"
4. JOIN this table with the EABLG table using INSTALLATION.
5. JOIN this table with the NET_MASTER table and the SALES_MASTER table using EQUNR.
6. Compare sales data and net data for these IDs:
    If sales data is there, take sales data and empty net data.
    Otherwise stay with net data and empty sales data.
7. Set ENTSCHEIDUNG to "MANDATORY_READING", for every EQUNR in EABL where both ISTABLART_TXT = "Maschinelle Schätzung - SAP" and the Record created on 2024 and ISTABLART_TXT = "Maschinelle Schätzung - SAP" and the Record created on 2023.
8. Export all MANDATORY_READING entries from the final table as CSV file named {tour}_MANDATORY_READING including the columns: `Meter Reading (MR) Doc. No.`, `HAUS`, `ANLAGE`, `ME_MA_ID`, `EQUNR`, `HOUSE_NUM1`, `ENTSCHEIDUNG`, `TITLE`, `FORENAME`, `SURNAME`, `STREET`, `STREETNO`, `POST_CODE1`, `CITY1`, `CITY2`, `EMAIL`
9. Send them to the the internal team to mandatory.reading@provider.com
10. Set ENTSCHEIDUNG to "DIRECT_MAIL", for every EQUNR in EABL where ISTABLART_TXT = "Ablesung durch Kunden - SAP" in 2024 and ISTABLART_TXT = "Ablesung durch Kunden - SAP" in 2023.
11. Export all DIRECT_MAIL entries from the final table as CSV file named {tour}_DIRECT_MAIL including the columns: `Meter Reading (MR) Doc. No.`, `HAUS`, `ANLAGE`, `ME_MA_ID`, `EQUNR`, `HOUSE_NUM1`, `ENTSCHEIDUNG`, `TITLE`, `FORENAME`, `SURNAME`, `STREET`, `STREETNO`, `POST_CODE1`, `CITY1`, `CITY2`, `EMAIL`
12. Upload the file to send_bulk_mail service for sending direct emails to all the customers.
13. Export the final data except MANDATORY_READING and DIRECT_MAIL entries from the table as CSV file named {tour} including the columns: `Meter Reading (MR) Doc. No.`, `HAUS`, `ANLAGE`, `ME_MA_ID`, `EQUNR`, `HOUSE_NUM1`, `ENTSCHEIDUNG`, `TITLE`, `FORENAME`, `SURNAME`, `STREET`, `STREETNO`, `POST_CODE1`, `CITY1`, `CITY2`, `EMAIL`
14. Send them to the EVU team to meter.readings@evu.com
15. Delete the temporary table after exporting the data.
""",
}

CLASSIC_RULES: dict[str, str] = {
    "base_rule": """I need you to process the meter readings for tour {tour}. Take all entries from the decision_table where the TOUR starts with "{tour}" and put them into a temporary table called {tour}_temp. Then add a new column called ENTSCHEIDUNG to that table.

For classifying each house: count how many equipment numbers (EQUNR) belong to each house (HAUS). If a house has fewer than 3 equipment numbers, set its ENTSCHEIDUNG to "KSA". Otherwise, set it to "EVU".

Next, enrich the data by joining it with the EABLG table using the INSTALLATION field, and also pull in customer contact details from both the net_master and sales_master tables using the EQUNR field. When both net and sales data exist for a customer, prefer the sales data and leave the net data fields empty. If only net data exists, keep that.

Export the final result as a CSV file named "{tour}" with these columns: `Meter Reading (MR) Doc. No.`, `HAUS`, `ANLAGE`, `ME_MA_ID`, `EQUNR`, `HOUSE_NUM1`, `ENTSCHEIDUNG`, `TITLE`, `FORENAME`, `SURNAME`, `STREET`, `STREETNO`, `POST_CODE1`, `CITY1`, `CITY2`, `EMAIL`.

Send the CSV to the EVU team at meter.readings@evu.com. After that, delete the temporary table.
""",
    "0_values": """I need you to process the meter readings for tour {tour}. Take all entries from the decision_table where the TOUR starts with "{tour}" and put them into a temporary table called {tour}_temp. Then add a new column called ENTSCHEIDUNG to that table.

For classifying each house: count how many equipment numbers (EQUNR) belong to each house (HAUS). If a house has fewer than 3 equipment numbers, set its ENTSCHEIDUNG to "KSA". Otherwise, set it to "EVU". Additionally, any house where HOUSE_NUM1 is 0 should always be set to "EVU" regardless of the count.

Next, enrich the data by joining it with the EABLG table using the INSTALLATION field, and also pull in customer contact details from both the net_master and sales_master tables using the EQUNR field. When both net and sales data exist for a customer, prefer the sales data and leave the net data fields empty. If only net data exists, keep that.

Export the final result as a CSV file named "{tour}" with these columns: `Meter Reading (MR) Doc. No.`, `HAUS`, `ANLAGE`, `ME_MA_ID`, `EQUNR`, `HOUSE_NUM1`, `ENTSCHEIDUNG`, `TITLE`, `FORENAME`, `SURNAME`, `STREET`, `STREETNO`, `POST_CODE1`, `CITY1`, `CITY2`, `EMAIL`.

Send the CSV to the EVU team at meter.readings@evu.com. After that, delete the temporary table.
""",
    "500_values": """I need you to process the meter readings for tour {tour}. Take all entries from the decision_table where the TOUR starts with "{tour}" and put them into a temporary table called {tour}_temp. Then add a new column called ENTSCHEIDUNG to that table.

For classifying each house: count how many equipment numbers (EQUNR) belong to each house (HAUS). If a house has fewer than 3 equipment numbers, set its ENTSCHEIDUNG to "KSA". Otherwise, set it to "EVU". There are two additional overrides: any house where HOUSE_NUM1 is 0 should be set to "EVU", and any house where HOUSE_NUM1 is in the range 500 to 599 should also be set to "EVU".

Next, enrich the data by joining it with the EABLG table using the INSTALLATION field, and also pull in customer contact details from both the net_master and sales_master tables using the EQUNR field. When both net and sales data exist for a customer, prefer the sales data and leave the net data fields empty. If only net data exists, keep that.

Export the final result as a CSV file named "{tour}" with these columns: `Meter Reading (MR) Doc. No.`, `HAUS`, `ANLAGE`, `ME_MA_ID`, `EQUNR`, `HOUSE_NUM1`, `ENTSCHEIDUNG`, `TITLE`, `FORENAME`, `SURNAME`, `STREET`, `STREETNO`, `POST_CODE1`, `CITY1`, `CITY2`, `EMAIL`.

Send the CSV to the EVU team at meter.readings@evu.com. After that, delete the temporary table.
""",
    "900_values": """I need you to process the meter readings for tour {tour}. Take all entries from the decision_table where the TOUR starts with "{tour}" and put them into a temporary table called {tour}_temp. Then add a new column called ENTSCHEIDUNG to that table.

For classifying each house: count how many equipment numbers (EQUNR) belong to each house (HAUS). If a house has fewer than 3 equipment numbers, set its ENTSCHEIDUNG to "KSA". Otherwise, set it to "EVU". There are three additional overrides that always result in "EVU": houses where HOUSE_NUM1 is 0, houses where HOUSE_NUM1 is between 500 and 599, and houses where HOUSE_NUM1 is between 900 and 999.

Next, enrich the data by joining it with the EABLG table using the INSTALLATION field, and also pull in customer contact details from both the net_master and sales_master tables using the EQUNR field. When both net and sales data exist for a customer, prefer the sales data and leave the net data fields empty. If only net data exists, keep that.

Export the final result as a CSV file named "{tour}" with these columns: `Meter Reading (MR) Doc. No.`, `HAUS`, `ANLAGE`, `ME_MA_ID`, `EQUNR`, `HOUSE_NUM1`, `ENTSCHEIDUNG`, `TITLE`, `FORENAME`, `SURNAME`, `STREET`, `STREETNO`, `POST_CODE1`, `CITY1`, `CITY2`, `EMAIL`.

Send the CSV to the EVU team at meter.readings@evu.com. After that, delete the temporary table.
""",
    "city_values": """I need you to process the meter readings for tour {tour}. Take all entries from the decision_table where the TOUR starts with "{tour}" and put them into a temporary table called {tour}_temp. Then add a new column called ENTSCHEIDUNG to that table.

For classifying each house: count how many equipment numbers (EQUNR) belong to each house (HAUS). If a house has fewer than 3 equipment numbers, set its ENTSCHEIDUNG to "KSA". Otherwise, set it to "EVU". There are four additional overrides that always result in "EVU": houses where HOUSE_NUM1 is 0, houses where HOUSE_NUM1 is between 500 and 599, houses where HOUSE_NUM1 is between 900 and 999, and houses located in the areas of Wiblingwerde or Breckerfeld.

Next, enrich the data by joining it with the EABLG table using the INSTALLATION field, and also pull in customer contact details from both the net_master and sales_master tables using the EQUNR field. When both net and sales data exist for a customer, prefer the sales data and leave the net data fields empty. If only net data exists, keep that.

Export the final result as a CSV file named "{tour}" with these columns: `Meter Reading (MR) Doc. No.`, `HAUS`, `ANLAGE`, `ME_MA_ID`, `EQUNR`, `HOUSE_NUM1`, `ENTSCHEIDUNG`, `TITLE`, `FORENAME`, `SURNAME`, `STREET`, `STREETNO`, `POST_CODE1`, `CITY1`, `CITY2`, `EMAIL`.

Send the CSV to the EVU team at meter.readings@evu.com. After that, delete the temporary table.
""",
    "extension_estimates": """I need you to process the meter readings for tour {tour}. Take all entries from the decision_table where the TOUR starts with "{tour}" and put them into a temporary table called {tour}_temp. Then add a new column called ENTSCHEIDUNG to that table.

For classifying each house: count how many equipment numbers (EQUNR) belong to each house (HAUS). If a house has fewer than 3 equipment numbers, set its ENTSCHEIDUNG to "KSA". Otherwise, set it to "EVU". There are four additional overrides that always result in "EVU": houses where HOUSE_NUM1 is 0, houses where HOUSE_NUM1 is between 500 and 599, houses where HOUSE_NUM1 is between 900 and 999, and houses located in the areas of Wiblingwerde or Breckerfeld.

Next, enrich the data by joining it with the EABLG table using the INSTALLATION field, and also pull in customer contact details from both the net_master and sales_master tables using the EQUNR field. When both net and sales data exist for a customer, prefer the sales data and leave the net data fields empty. If only net data exists, keep that.

There is one more classification to handle: check the EABL table for equipment numbers that had "Maschinelle Schätzung - SAP" as ISTABLART_TXT in both 2023 and 2024 (based on Record created on). For those, set ENTSCHEIDUNG to "MANDATORY_READING".

Export the MANDATORY_READING entries separately as a CSV file named "{tour}_MANDATORY_READING" with these columns: `Meter Reading (MR) Doc. No.`, `HAUS`, `ANLAGE`, `ME_MA_ID`, `EQUNR`, `HOUSE_NUM1`, `ENTSCHEIDUNG`, `TITLE`, `FORENAME`, `SURNAME`, `STREET`, `STREETNO`, `POST_CODE1`, `CITY1`, `CITY2`, `EMAIL`. Send that file to the internal team at mandatory.reading@provider.com.

Then export all remaining entries (everything except MANDATORY_READING) as a CSV file named "{tour}" with the same columns. Send that to the EVU team at meter.readings@evu.com. After that, delete the temporary table.
""",
    "extension_mail": """I need you to process the meter readings for tour {tour}. Take all entries from the decision_table where the TOUR starts with "{tour}" and put them into a temporary table called {tour}_temp. Then add a new column called ENTSCHEIDUNG to that table.

For classifying each house: count how many equipment numbers (EQUNR) belong to each house (HAUS). If a house has fewer than 3 equipment numbers, set its ENTSCHEIDUNG to "KSA". Otherwise, set it to "EVU". There are four additional overrides that always result in "EVU": houses where HOUSE_NUM1 is 0, houses where HOUSE_NUM1 is between 500 and 599, houses where HOUSE_NUM1 is between 900 and 999, and houses located in the areas of Wiblingwerde or Breckerfeld.

Next, enrich the data by joining it with the EABLG table using the INSTALLATION field, and also pull in customer contact details from both the net_master and sales_master tables using the EQUNR field. When both net and sales data exist for a customer, prefer the sales data and leave the net data fields empty. If only net data exists, keep that.

There are two more classifications to handle. First, check the EABL table for equipment numbers that had "Maschinelle Schätzung - SAP" as ISTABLART_TXT in both 2023 and 2024 (based on Record created on). For those, set ENTSCHEIDUNG to "MANDATORY_READING". Export them as a CSV file named "{tour}_MANDATORY_READING" with the columns listed below and send it to the internal team at mandatory.reading@provider.com.

Second, check the EABL table for equipment numbers that had "Ablesung durch Kunden - SAP" as ISTABLART_TXT in both 2023 and 2024. For those, set ENTSCHEIDUNG to "DIRECT_MAIL". Export them as a CSV file named "{tour}_DIRECT_MAIL" with the columns listed below and upload that file to the send_bulk_mail service so emails go out to all those customers directly.

Finally, export all remaining entries (excluding MANDATORY_READING and DIRECT_MAIL) as a CSV file named "{tour}" with the same columns. Send that to the EVU team at meter.readings@evu.com. After that, delete the temporary table.

The columns for all exports are: `Meter Reading (MR) Doc. No.`, `HAUS`, `ANLAGE`, `ME_MA_ID`, `EQUNR`, `HOUSE_NUM1`, `ENTSCHEIDUNG`, `TITLE`, `FORENAME`, `SURNAME`, `STREET`, `STREETNO`, `POST_CODE1`, `CITY1`, `CITY2`, `EMAIL`.
""",
}

BPMN_RULES: dict[str, dict[str, str]] = {
    "base_rule": {
        "J09A": "Use the generate_process_rule_from_bpmn tool to generate a process rule from the following BPMN file: bpmn/J09A_base_rule.bpmn and store it in the database as J09A",
        "J09B": "Use the generate_process_rule_from_bpmn tool to generate a process rule from the following BPMN file: bpmn/J09B_base_rule.bpmn and store it in the database as J09B",
        "J09C": "Use the generate_process_rule_from_bpmn tool to generate a process rule from the following BPMN file: bpmn/J09C_base_rule.bpmn and store it in the database as J09C",
        "J09D": "Use the generate_process_rule_from_bpmn tool to generate a process rule from the following BPMN file: bpmn/J09D_base_rule.bpmn and store it in the database as J09D",
    },
    "0_values": {
        "J09A": "Use the generate_process_rule_from_bpmn tool to generate a process rule from the following BPMN file: bpmn/J09A_0_values.bpmn and store it in the database as J09A",
        "J09B": "Use the generate_process_rule_from_bpmn tool to generate a process rule from the following BPMN file: bpmn/J09B_0_values.bpmn and store it in the database as J09B",
        "J09C": "Use the generate_process_rule_from_bpmn tool to generate a process rule from the following BPMN file: bpmn/J09C_0_values.bpmn and store it in the database as J09C",
        "J09D": "Use the generate_process_rule_from_bpmn tool to generate a process rule from the following BPMN file: bpmn/J09D_0_values.bpmn and store it in the database as J09D",
    },
    "500_values": {
        "J09A": "Use the generate_process_rule_from_bpmn tool to generate a process rule from the following BPMN file: bpmn/J09A_500_values.bpmn and store it in the database as J09A",
        "J09B": "Use the generate_process_rule_from_bpmn tool to generate a process rule from the following BPMN file: bpmn/J09B_500_values.bpmn and store it in the database as J09B",
        "J09C": "Use the generate_process_rule_from_bpmn tool to generate a process rule from the following BPMN file: bpmn/J09C_500_values.bpmn and store it in the database as J09C",
        "J09D": "Use the generate_process_rule_from_bpmn tool to generate a process rule from the following BPMN file: bpmn/J09D_500_values.bpmn and store it in the database as J09D",
    },
    "900_values": {
        "J09A": "Use the generate_process_rule_from_bpmn tool to generate a process rule from the following BPMN file: bpmn/J09A_900_values.bpmn and store it in the database as J09A",
        "J09B": "Use the generate_process_rule_from_bpmn tool to generate a process rule from the following BPMN file: bpmn/J09B_900_values.bpmn and store it in the database as J09B",
        "J09C": "Use the generate_process_rule_from_bpmn tool to generate a process rule from the following BPMN file: bpmn/J09C_900_values.bpmn and store it in the database as J09C",
        "J09D": "Use the generate_process_rule_from_bpmn tool to generate a process rule from the following BPMN file: bpmn/J09D_900_values.bpmn and store it in the database as J09D",
    },
    "city_values": {
        "J09A": "Use the generate_process_rule_from_bpmn tool to generate a process rule from the following BPMN file: bpmn/J09A_city_values.bpmn and store it in the database as J09A",
        "J09B": "Use the generate_process_rule_from_bpmn tool to generate a process rule from the following BPMN file: bpmn/J09B_city_values.bpmn and store it in the database as J09B",
        "J09C": "Use the generate_process_rule_from_bpmn tool to generate a process rule from the following BPMN file: bpmn/J09C_city_values.bpmn and store it in the database as J09C",
        "J09D": "Use the generate_process_rule_from_bpmn tool to generate a process rule from the following BPMN file: bpmn/J09D_city_values.bpmn and store it in the database as J09D",
    },
    "extension_estimates": {
        "J09A": "Use the generate_process_rule_from_bpmn tool to generate a process rule from the following BPMN file: bpmn/J09A_extension_estimates.bpmn and store it in the database as J09A",
        "J09B": "Use the generate_process_rule_from_bpmn tool to generate a process rule from the following BPMN file: bpmn/J09B_extension_estimates.bpmn and store it in the database as J09B",
        "J09C": "Use the generate_process_rule_from_bpmn tool to generate a process rule from the following BPMN file: bpmn/J09C_extension_estimates.bpmn and store it in the database as J09C",
        "J09D": "Use the generate_process_rule_from_bpmn tool to generate a process rule from the following BPMN file: bpmn/J09D_extension_estimates.bpmn and store it in the database as J09D",
    },
    "extension_mail": {
        "J09A": "Use the generate_process_rule_from_bpmn tool to generate a process rule from the following BPMN file: bpmn/J09A_extension_mail.bpmn and store it in the database as J09A",
        "J09B": "Use the generate_process_rule_from_bpmn tool to generate a process rule from the following BPMN file: bpmn/J09B_extension_mail.bpmn and store it in the database as J09B",
        "J09C": "Use the generate_process_rule_from_bpmn tool to generate a process rule from the following BPMN file: bpmn/J09C_extension_mail.bpmn and store it in the database as J09C",
        "J09D": "Use the generate_process_rule_from_bpmn tool to generate a process rule from the following BPMN file: bpmn/J09D_extension_mail.bpmn and store it in the database as J09D",
    },
}
