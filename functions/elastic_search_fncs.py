import pandas as pd
import numpy as np
import re
from elasticsearch import Elasticsearch

class ElasticSearchFncs:

    def __init__(self):
        # Details of the dataset
        self.db_host = 'https://athena.london.gov.uk'
        self.db_user = 'odbc_readonly'
        self.db_pass = 'odbc_readonly'
        self.db_port = '10099'
        self.db_name = 'gla-ldd-external'
        self.index = 'applications'

        # Creates connection to the dataset
        self.es = Elasticsearch(
            [f"{self.db_host}:{self.db_port}"],
            basic_auth=(self.db_user, self.db_pass), 
            verify_certs=True,
            ca_certs='../athena_es_full_chain.crt'
        )



    def check_connection(self):
        """Check connection to the Elastic Search database."""
        if self.es.ping():
            print("Connected to Elastic Search")
        else:
            print("Could not connect to Elastic Search")

    

    def _scroll_and_retrieve(self, query):
        # Initialise scrolling
        response = self.es.search(index=self.index, body=query, scroll="2m", size=10000)
        scroll_id = response['_scroll_id']
        hits = response['hits']['hits']
        
        all_hits = []
        all_hits.extend(hits)

        # Keep scrolling until no hits are returned
        while len(hits) > 0:
            response = self.es.scroll(scroll_id=scroll_id, scroll="2m")
            scroll_id = response['_scroll_id']
            hits = response['hits']['hits']
            all_hits.extend(hits)

        # Normalize the results into a DataFrame
        df = pd.json_normalize(all_hits)

        return df 
    


    def _year_bucket_and_retrieve(self, query):
        # Execute the query
        response = self.es.search(index=self.index, body=query)

        # Process the response into a DataFrame
        buckets = response["aggregations"]["applications_by_year"]["buckets"]
        data = []

        for bucket in buckets:
            year = bucket["key_as_string"][-4:]  # Extract the year from the timestamp
            range_data = {}
            for range_bucket in bucket["unit_ranges"]["buckets"]:
                range_key = f"{int(range_bucket.get('from', 0))}-{int(range_bucket.get('to', 0))}"
                range_total_units = range_bucket["total_units"]["value"]
                range_data[range_key] = range_total_units
            range_data["year"] = year
            data.append(range_data)

        # Create a DataFrame, filling missing ranges with 0
        df = pd.DataFrame(data).fillna(0)

        return df



    def res_units_x_query(self, min_res_units, since_year="01/01/2014", to_year="now"):
        """Elastic Search query to filter applications with a minimum number of proposed residential units.

        Args:
            es (obj): Elastic Search object.
            min_res_units (int): Minimum number of proposed residential units.
            since_year (str, optional): Initial date of the search. Defaults to "01/01/2014".

        Returns:
            dataframe: Dataframe with the filtered applications.
        """

        query = {
        "query": {
            "bool": {
            "must": [
                {
                "range": {
                    "application_details.residential_details.total_no_proposed_residential_units": {
                    "gte": min_res_units
                    }
                }
                },
                {
                "range": {
                    "valid_date": {
                    "gte": since_year,
                    "lt": to_year  
                    }
                }
                }
            ],
            "must_not": [
                {
                "term": {
                    "status.keyword": "Superseded"
                }
                }
            ]
            }
        },
        "_source": [
            "borough",
            "decision_date",
            "valid_date",
            "decision",
            "actual_completion_date",
            "application_details.residential_details.total_no_proposed_residential_units",
            "application_details.residential_details.site_area",
            "application_details.affordable_housing_fast_track",
            "application_details.residential_details.total_no_affordable_units",
            "application_details.residential_details.total_no_proposed_residential_units_discount_market_rent",
            "application_details.residential_details.total_no_proposed_residential_units_discount_market_rent_charged_at_london_rents",
            "application_details.residential_details.total_no_proposed_residential_units_discount_market_sale",
            "application_details.residential_details.total_no_proposed_residential_units_intermediate",
            "application_details.residential_details.total_no_proposed_residential_units_london_affordable_rent",
            "application_details.residential_details.total_no_proposed_residential_units_london_living_rent",
            "application_details.residential_details.total_no_proposed_residential_units_london_shared_ownership",
            "application_details.residential_details.total_no_proposed_residential_units_market_for_rent",
            "application_details.residential_details.total_no_proposed_residential_units_market_for_sale",
            "application_details.residential_details.total_no_proposed_residential_units_self_build_and_custom_build",
            "application_details.residential_details.total_no_proposed_residential_units_shared_equity",
            "application_details.residential_details.total_no_proposed_residential_units_social_rent",
            "application_details.residential_details.total_no_proposed_residential_units_starter_homes",
            "status",
            "id",
            "pp_id",
            "lpa_name",
            "lpa_app_no",
            "site_name",
            "site_number",
            "street_name",
            "uprn",
            "polygon",
            "wgs84_polygon",
            "wgs84_polygon.coordinates",
            "polygon.coordinates",
            "postcode"
        ]
        }

        df = self._scroll_and_retrieve(query)

        return df



    def social_units_x_query(self, min_social_units, since_year="01/01/2014", to_year="now"):
        """Elastic Search query to filter applications with a minimum number of proposed social rent units.

        Args:
            es (obj): Elastic Search object.
            min_social_units (int): Minimum number of proposed residential units.
            since_year (str, optional): Initial date of the search. Defaults to "01/01/2014".

        Returns:
            dataframe: Dataframe with the filtered applications.
        """

        query = {
        "query": {
            "bool": {
            "must": [
                {
                "range": {
                    "application_details.residential_details.total_no_proposed_residential_units_social_rent": { # this line specifies only applications which add council housing  
                    "gte": min_social_units
                    }
                }
                },
                {
                "range": {
                    "valid_date": {
                    "gte": since_year,
                    "lt": to_year  
                    }
                }
                }
            ],
            "must_not": [
                {
                "term": {
                    "status.keyword": "Superseded"
                }
                }
            ]
            }
        },
        "_source": [
            "borough",
            "decision_date",
            "valid_date",
            "decision",
            "actual_completion_date",
            "application_details.residential_details.total_no_proposed_residential_units",
            "application_details.residential_details.site_area",
            "application_details.affordable_housing_fast_track",
            "application_details.residential_details.total_no_affordable_units",
            "application_details.residential_details.total_no_proposed_residential_units_discount_market_rent",
            "application_details.residential_details.total_no_proposed_residential_units_discount_market_rent_charged_at_london_rents",
            "application_details.residential_details.total_no_proposed_residential_units_discount_market_sale",
            "application_details.residential_details.total_no_proposed_residential_units_intermediate",
            "application_details.residential_details.total_no_proposed_residential_units_london_affordable_rent",
            "application_details.residential_details.total_no_proposed_residential_units_london_living_rent",
            "application_details.residential_details.total_no_proposed_residential_units_london_shared_ownership",
            "application_details.residential_details.total_no_proposed_residential_units_market_for_rent",
            "application_details.residential_details.total_no_proposed_residential_units_market_for_sale",
            "application_details.residential_details.total_no_proposed_residential_units_self_build_and_custom_build",
            "application_details.residential_details.total_no_proposed_residential_units_shared_equity",
            "application_details.residential_details.total_no_proposed_residential_units_social_rent",
            "application_details.residential_details.total_no_proposed_residential_units_starter_homes",
            "status",
            "id",
            "pp_id",
            "lpa_name",
            "lpa_app_no",
            "site_name",
            "site_number",
            "street_name",
            "postcode",
            "uprn",
            "polygon",
            "wgs84_polygon",
            "description"
        ]
        }

        df = self._scroll_and_retrieve(query)

        return df
    


    def get_approved_units_by_ranges(self, since_year="01/01/2014", to_year="now"):
        """
        Get the number of permitted applications and total number of residential units applied for by year.
        These are grouped into ranges of residential units.

        Args:
            es (obj): elasticsearch object.
            index (str, optional): elasticsearch index. Defaults to "applications".
            since_year (str, optional): Include applications with decision made from this date onwards. Defaults to "01/01/2010".

        Returns:
            dataframe: Dataframe with the number of applications and total number of residential units by year and range.
        """

        # aggregation includes the from value and excludes the to value for each range
        ranges = [
            {"from": 1, "to": 2},      # 1
            {"from": 2, "to": 10},     # 2-9
            {"from": 10, "to": 50},    # 10-49
            {"from": 50, "to": 100},   # 50-99
            {"from": 100, "to": 150},  # 100-149
            {"from": 150, "to": 200},  # 150-199
            {"from": 200, "to": 250},  # 200-249
            {"from": 250, "to": 300},  # 250-299
            {"from": 300, "to": 350},  # 300-349
            {"from": 350, "to": 400},  # 350-399
            {"from": 400, "to": 450},  # 400-449
            {"from": 450, "to": 500},  # 450-499
            {"from": 500},             # 500+
        ]

        # Define the query with range aggregations
        query = {
            "query": {
                "bool": {
                "must": [
                    {
                    "range": {
                        "decision_date": {
                        "gte": since_year, 
                        "lte": to_year
                        }
                    }
                    },
                    {
                    "range": {
                        "application_details.residential_details.total_no_proposed_residential_units": {
                            "gte": 1
                        }
                    }
                    }
                ],
                "must_not": [
                    {
                    "term": {
                        "status.keyword": "Superseded"
                    }
                    }
                ],
                "should": [
                    {
                    "query_string": {
                        "query": "(Approved) OR (Not required) OR (Allowed)",
                        "fields": ["decision", "status"]
                    },
                    },
                    {
                    "exists":{
                        "field" : "actual_completion_date"
                    }
                    }
                ],
                "minimum_should_match" : 1,
                }
            },
            "size": 0,  # No documents, only aggregation results
            "aggs": {
                "applications_by_year": {
                    "date_histogram": {
                        "field": "valid_date",
                        "calendar_interval": "year"
                    },
                    "aggs": {
                        "unit_ranges": {
                            "range": {
                                "field": "application_details.residential_details.total_no_proposed_residential_units",
                                "ranges": ranges
                            },
                            "aggs": {
                                "total_units": {
                                    "sum": {
                                        "field": "application_details.residential_details.total_no_proposed_residential_units"
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        df = self._year_bucket_and_retrieve(query)

        return df