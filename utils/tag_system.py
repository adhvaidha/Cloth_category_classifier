"""
Comprehensive Tag System for Personalized Outfit Recommendations

This module defines the expanded tag system that captures user preferences
and influences outfit recommendations with fine-grained control.
"""

from typing import Dict, List, Optional, Set, Any
from enum import Enum


class OccasionTag(str, Enum):
    """Occasion tags matching API Literal types"""
    CASUAL = "casual"
    BUSINESS = "business"
    FORMAL = "formal"
    SEMI_FORMAL = "semi_formal"
    BUSINESS_CASUAL = "business_casual"
    COCKTAIL = "cocktail"
    WEDDING = "wedding"
    PARTY = "party"
    DATE = "date"
    SPORT = "sport"
    WORKOUT = "workout"
    TRAVEL = "travel"
    BEACH = "beach"
    OUTDOOR = "outdoor"
    NIGHT_OUT = "night_out"
    BRUNCH = "brunch"
    DINNER = "dinner"
    MEETING = "meeting"
    INTERVIEW = "interview"
    CULTURAL = "cultural"
    TRADITIONAL = "traditional"


class WeatherTag(str, Enum):
    """Weather tags matching API Literal types"""
    ANY = "any"
    HOT = "hot"
    WARM = "warm"
    MILD = "mild"
    COOL = "cool"
    COLD = "cold"
    FREEZING = "freezing"
    RAIN = "rain"
    SNOW = "snow"
    WINDY = "windy"
    HUMID = "humid"
    SUNNY = "sunny"
    CLOUDY = "cloudy"


class StyleTag(str, Enum):
    """Style tags matching API Literal types (outfit_style)"""
    CASUAL = "casual"
    SMART_CASUAL = "smart_casual"
    FORMAL = "formal"
    SPORTY = "sporty"
    ATHLETIC = "athletic"
    STREETWEAR = "streetwear"
    MINIMALIST = "minimalist"
    CLASSIC = "classic"
    MODERN = "modern"
    ELEGANT = "elegant"
    SOPHISTICATED = "sophisticated"
    TRADITIONAL = "traditional"
    ETHNIC = "ethnic"


class ColorPreferenceTag(str, Enum):
    """Color preference tags matching API Literal types"""
    NEUTRAL = "neutral"
    MONOCHROMATIC = "monochromatic"
    COMPLEMENTARY = "complementary"
    BOLD = "bold"
    SUBTLE = "subtle"
    BRIGHT = "bright"
    MUTED = "muted"
    PASTEL = "pastel"
    DARK = "dark"
    LIGHT = "light"
    EARTH_TONES = "earth_tones"
    JEWEL_TONES = "jewel_tones"
    BLACK_WHITE = "black_white"
    NAVY_WHITE = "navy_white"
    COLORFUL = "colorful"
    MINIMAL_COLOR = "minimal_color"


class FitPreferenceTag(str, Enum):
    """Fit preference tags matching API Literal types"""
    FITTED = "fitted"
    LOOSE = "loose"
    OVERSIZED = "oversized"
    RELAXED = "relaxed"
    COMFORTABLE = "comfortable"
    STRUCTURED = "structured"
    FLOWY = "flowy"
    TAILORED = "tailored"
    ATHLETIC_FIT = "athletic_fit"
    REGULAR_FIT = "regular_fit"


class MaterialPreferenceTag(str, Enum):
    """Material preference tags matching API Literal types"""
    COTTON = "cotton"
    LINEN = "linen"
    SILK = "silk"
    WOOL = "wool"
    CASHMERE = "cashmere"
    DENIM = "denim"
    LEATHER = "leather"
    BREATHABLE = "breathable"
    WATERPROOF = "waterproof"
    MOISTURE_WICKING = "moisture_wicking"
    SUSTAINABLE = "sustainable"


class BudgetTag(str, Enum):
    """Budget preference tags matching API Literal types"""
    LUXURY = "luxury"
    PREMIUM = "premium"
    MID_RANGE = "mid_range"
    AFFORDABLE = "affordable"
    BUDGET = "budget"
    VALUE = "value"


class AgeGroupTag(str, Enum):
    """Age group tags for age-appropriate styling"""
    TEEN = "teen"
    YOUNG_ADULT = "young_adult"
    ADULT = "adult"
    MATURE = "mature"
    SENIOR = "senior"
    ALL_AGES = "all_ages"


class GenderTag(str, Enum):
    """Gender tags for gender-specific recommendations"""
    MALE = "male"
    FEMALE = "female"
    NON_BINARY = "non_binary"
    UNISEX = "unisex"
    ANY = "any"


class SeasonTag(str, Enum):
    """Season tags for seasonal styling"""
    SPRING = "spring"
    SUMMER = "summer"
    FALL = "fall"
    AUTUMN = "autumn"
    WINTER = "winter"
    YEAR_ROUND = "year_round"
    TRANSITIONAL = "transitional"


class TimeOfDayTag(str, Enum):
    """Time of day tags for time-appropriate styling"""
    MORNING = "morning"
    AFTERNOON = "afternoon"
    EVENING = "evening"
    NIGHT = "night"
    ALL_DAY = "all_day"


class PersonalStyleTag(str, Enum):
    """Personal style preference tags"""
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    BOLD = "bold"
    EXPERIMENTAL = "experimental"
    TRADITIONAL = "traditional"
    TRENDY = "trendy"
    TIMELESS = "timeless"
    FASHION_FORWARD = "fashion_forward"
    CLASSIC = "classic"
    ECLECTIC = "eclectic"


class TagProcessor:
    """
    Processes and prioritizes tags to influence outfit recommendations.
    Handles tag conflicts, weights, and generates context-aware preferences.
    """
    
    def __init__(self):
        self.tag_weights = self._initialize_tag_weights()
        self.tag_conflicts = self._initialize_tag_conflicts()
        self.tag_synergies = self._initialize_tag_synergies()
    
    def _initialize_tag_weights(self) -> Dict[str, float]:
        """Initialize default weights for different tag categories"""
        return {
            "occasion": 1.0,          # Highest priority
            "weather": 0.9,           # Very high priority
            "style": 0.8,             # High priority
            "color_preference": 0.7,  # Medium-high priority
            "fit_preference": 0.6,   # Medium priority
            "material_preference": 0.5,  # Medium priority
            "season": 0.4,           # Lower priority
            "time_of_day": 0.3,      # Lower priority
            "budget": 0.2,            # Lower priority
            "age_group": 0.1,         # Lower priority
            "gender": 0.1,            # Lower priority
            "personal_style": 0.5,    # Medium priority
        }
    
    def _initialize_tag_conflicts(self) -> Dict[str, List[str]]:
        """Define conflicting tags that should not be used together"""
        return {
            "hot": ["cold", "freezing", "snow"],
            "cold": ["hot", "warm"],
            "formal": ["casual", "sporty"],
            "sporty": ["formal", "elegant"],
            "budget": ["luxury"],
            "loose": ["fitted", "tight"],
            "tight": ["loose", "oversized"],
        }
    
    def _initialize_tag_synergies(self) -> Dict[str, List[str]]:
        """Define tag synergies that work well together"""
        return {
            "formal": ["elegant", "sophisticated", "tailored"],
            "casual": ["comfortable", "relaxed", "practical"],
            "sporty": ["athletic", "comfortable", "moisture_wicking"],
            "hot": ["breathable", "cotton", "linen", "light"],
            "cold": ["wool", "cashmere", "warm", "structured"],
            "rain": ["waterproof", "practical", "boot"],
            "business": ["professional", "tailored", "neutral"],
            "date": ["elegant", "romantic", "fitted"],
            "party": ["glamorous", "bold", "colorful"],
        }
    
    def process_tags(self, tags: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process and prioritize tags, resolving conflicts and applying synergies.
        
        Args:
            tags: Dictionary of tag categories and their values
            
        Returns:
            Processed tags with weights, resolved conflicts, and applied synergies
        """
        processed = {
            "primary_tags": {},
            "secondary_tags": {},
            "weights": {},
            "preferences": {},
            "constraints": {}
        }
        
        # Extract and validate tags (support both "style" and "outfit_style" for backward compatibility)
        occasion = tags.get("occasion", "casual")
        weather = tags.get("weather", "any")
        outfit_style = tags.get("outfit_style") or tags.get("style", "casual")  # Support both keys
        color_preference = tags.get("color_preference", None)
        fit_preference = tags.get("fit_preference", None)
        material_preference = tags.get("material_preference", None)
        season = tags.get("season", None)
        time_of_day = tags.get("time_of_day", None)
        budget = tags.get("budget", None)
        age_group = tags.get("age_group", None)
        gender = tags.get("gender", None)
        personal_style = tags.get("personal_style", None)
        
        # Primary tags (high influence)
        processed["primary_tags"] = {
            "occasion": occasion,
            "weather": weather,
            "outfit_style": outfit_style  # Use outfit_style consistently
        }
        
        # Secondary tags (medium influence)
        secondary = {}
        if color_preference:
            secondary["color_preference"] = color_preference
        if fit_preference:
            secondary["fit_preference"] = fit_preference
        if material_preference:
            secondary["material_preference"] = material_preference
        if personal_style:
            secondary["personal_style"] = personal_style
        processed["secondary_tags"] = secondary
        
        # Tertiary tags (lower influence)
        tertiary = {}
        if season:
            tertiary["season"] = season
        if time_of_day:
            tertiary["time_of_day"] = time_of_day
        if budget:
            tertiary["budget"] = budget
        if age_group:
            tertiary["age_group"] = age_group
        if gender:
            tertiary["gender"] = gender
        processed["tertiary_tags"] = tertiary
        
        # Resolve conflicts
        processed["resolved_tags"] = self._resolve_conflicts(
            processed["primary_tags"], processed["secondary_tags"]
        )
        
        # Apply synergies
        processed["synergies"] = self._apply_synergies(
            processed["resolved_tags"]
        )
        
        # Calculate weights
        processed["weights"] = self._calculate_weights(
            processed["primary_tags"],
            processed["secondary_tags"],
            processed["tertiary_tags"]
        )
        
        # Generate preferences
        processed["preferences"] = self._generate_preferences(
            processed["resolved_tags"],
            processed["synergies"]
        )
        
        # Generate constraints
        processed["constraints"] = self._generate_constraints(
            processed["resolved_tags"]
        )
        
        return processed
    
    def _resolve_conflicts(self, primary: Dict, secondary: Dict) -> Dict:
        """Resolve tag conflicts by prioritizing primary tags"""
        resolved = {**primary, **secondary}
        
        for tag, conflicts in self.tag_conflicts.items():
            if tag in resolved.values():
                for conflict in conflicts:
                    if conflict in resolved.values():
                        # Remove conflict from secondary tags, keep primary
                        for key, value in list(resolved.items()):
                            if value == conflict and key not in primary:
                                del resolved[key]
        
        return resolved
    
    def _apply_synergies(self, tags: Dict) -> List[str]:
        """Apply tag synergies to enhance recommendations"""
        synergies = []
        tag_values = list(tags.values())
        
        for tag, synergy_list in self.tag_synergies.items():
            if tag in tag_values:
                synergies.extend(synergy_list)
        
        return list(set(synergies))  # Remove duplicates
    
    def _calculate_weights(self, primary: Dict, secondary: Dict, tertiary: Dict) -> Dict:
        """Calculate weights for different tag categories"""
        weights = {}
        
        for category in primary:
            weights[category] = self.tag_weights.get(category, 1.0)
        
        for category in secondary:
            weights[category] = self.tag_weights.get(category, 0.5)
        
        for category in tertiary:
            weights[category] = self.tag_weights.get(category, 0.2)
        
        return weights
    
    def _generate_preferences(self, tags: Dict, synergies: List[str]) -> Dict:
        """Generate preference dictionary for recommendation engine"""
        preferences = {
            "preferred_categories": [],
            "color_palette": [],
            "material_preferences": [],
            "fit_preferences": [],
            "style_keywords": []
        }
        
        # Map tags to preferences (support both "style" and "outfit_style")
        style_value = tags.get("outfit_style") or tags.get("style", "casual")
        style_mapping = {
            "formal": ["blazer", "jacket", "dress shirt", "dress pant", "oxford"],
            "casual": ["tshirt", "jean", "sneaker", "hoodie"],
            "sporty": ["athletic shirt", "jogger", "running shoe", "tank"],
            "athletic": ["athletic shirt", "jogger", "running shoe", "tank"],
            "business": ["shirt", "pants", "loafer", "blazer"],
            "smart_casual": ["shirt", "chino", "loafer", "blazer"],
            "traditional": ["kameez", "kurta", "shalwar", "peshawari"],
            "ethnic": ["kameez", "kurta", "shalwar", "peshawari"],
            "elegant": ["blazer", "dress shirt", "dress pant", "oxford"],
            "sophisticated": ["blazer", "dress shirt", "dress pant", "oxford"],
            "minimalist": ["tshirt", "jean", "sneaker", "hoodie"],
            "classic": ["shirt", "pants", "loafer", "blazer"],
            "modern": ["tshirt", "jean", "sneaker", "hoodie"],
            "streetwear": ["tshirt", "jean", "sneaker", "hoodie"]
        }
        
        if style_value in style_mapping:
            preferences["preferred_categories"].extend(style_mapping[style_value])
        
        # Add synergies as style keywords
        preferences["style_keywords"].extend(synergies)
        
        return preferences
    
    def _generate_constraints(self, tags: Dict) -> Dict:
        """Generate constraints based on tags"""
        constraints = {
            "min_items": 2,
            "max_items": 6,
            "accessory_limit": 3,
            "requires_outerwear": False,
            "requires_formal": False
        }
        
        # Occasion-based constraints
        if tags.get("occasion") == "formal":
            constraints["min_items"] = 4
            constraints["max_items"] = 5
            constraints["requires_outerwear"] = True
            constraints["requires_formal"] = True
            constraints["accessory_limit"] = 4
        
        elif tags.get("occasion") == "business":
            constraints["min_items"] = 3
            constraints["max_items"] = 4
            constraints["accessory_limit"] = 3
        
        elif tags.get("occasion") == "sport":
            constraints["min_items"] = 2
            constraints["max_items"] = 3
            constraints["accessory_limit"] = 1
        
        # Weather-based constraints
        if tags.get("weather") in ["cold", "freezing"]:
            constraints["requires_outerwear"] = True
        
        return constraints


def get_all_tag_options() -> Dict[str, List[str]]:
    """Get all available tag options for UI/API"""
    return {
        "occasion": [tag.value for tag in OccasionTag],
        "weather": [tag.value for tag in WeatherTag],
        "outfit_style": [tag.value for tag in StyleTag],  # Use outfit_style to match API
        "style": [tag.value for tag in StyleTag],  # Keep for backward compatibility
        "color_preference": [tag.value for tag in ColorPreferenceTag],
        "fit_preference": [tag.value for tag in FitPreferenceTag],
        "material_preference": [tag.value for tag in MaterialPreferenceTag],
        "season": [tag.value for tag in SeasonTag],
        "time_of_day": [tag.value for tag in TimeOfDayTag],
        "budget": [tag.value for tag in BudgetTag],
        "age_group": [tag.value for tag in AgeGroupTag],
        "gender": [tag.value for tag in GenderTag],
        "personal_style": [tag.value for tag in PersonalStyleTag],
    }


def validate_tags(tags: Dict[str, Any]) -> tuple[bool, List[str]]:
    """Validate tags and return (is_valid, errors)"""
    errors = []
    all_options = get_all_tag_options()
    
    for category, value in tags.items():
        if category not in all_options:
            errors.append(f"Unknown tag category: {category}")
            continue
        
        if value not in all_options[category]:
            errors.append(f"Invalid value '{value}' for category '{category}'. Valid options: {all_options[category]}")
    
    return len(errors) == 0, errors
