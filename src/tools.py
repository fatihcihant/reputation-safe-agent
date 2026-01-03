"""Mock tools for the sub-agents."""

from src.models import Order, Product


# Mock database
MOCK_ORDERS: dict[str, Order] = {
    "ORD-001": Order(
        order_id="ORD-001",
        status="shipped",
        items=[{"name": "Wireless Headphones", "qty": 1, "price": 149.99}],
        total=149.99,
        shipping_address="123 Main St, Istanbul, TR",
        tracking_number="TRK123456789"
    ),
    "ORD-002": Order(
        order_id="ORD-002",
        status="processing",
        items=[
            {"name": "USB-C Cable", "qty": 2, "price": 19.99},
            {"name": "Phone Case", "qty": 1, "price": 29.99}
        ],
        total=69.97,
        shipping_address="456 Oak Ave, Ankara, TR",
        tracking_number=None
    ),
    "ORD-003": Order(
        order_id="ORD-003",
        status="delivered",
        items=[{"name": "Laptop Stand", "qty": 1, "price": 89.99}],
        total=89.99,
        shipping_address="789 Pine Rd, Izmir, TR",
        tracking_number="TRK987654321"
    ),
}

MOCK_PRODUCTS: dict[str, Product] = {
    "PROD-001": Product(
        product_id="PROD-001",
        name="Wireless Headphones Pro",
        price=149.99,
        description="Premium wireless headphones with active noise cancellation",
        category="Electronics",
        in_stock=True,
        specs={"battery_life": "30 hours", "driver_size": "40mm", "bluetooth": "5.3"}
    ),
    "PROD-002": Product(
        product_id="PROD-002",
        name="USB-C Fast Charging Cable",
        price=19.99,
        description="2m braided USB-C cable with 100W fast charging support",
        category="Accessories",
        in_stock=True,
        specs={"length": "2m", "max_power": "100W", "material": "braided nylon"}
    ),
    "PROD-003": Product(
        product_id="PROD-003",
        name="Ergonomic Laptop Stand",
        price=89.99,
        description="Adjustable aluminum laptop stand with cooling design",
        category="Accessories",
        in_stock=False,
        specs={"material": "aluminum", "adjustable_height": "15-25cm", "max_weight": "10kg"}
    ),
    "PROD-004": Product(
        product_id="PROD-004",
        name="Mechanical Keyboard RGB",
        price=129.99,
        description="Full-size mechanical keyboard with RGB backlighting",
        category="Electronics",
        in_stock=True,
        specs={"switch_type": "Cherry MX Red", "layout": "Full-size", "connectivity": "USB/Wireless"}
    ),
}


class OrderTools:
    """Tools for order-related operations."""
    
    @staticmethod
    def get_order(order_id: str) -> dict | None:
        order = MOCK_ORDERS.get(order_id.upper())
        if order:
            return {
                "order_id": order.order_id,
                "status": order.status,
                "items": order.items,
                "total": order.total,
                "shipping_address": order.shipping_address,
                "tracking_number": order.tracking_number
            }
        return None
    
    @staticmethod
    def get_order_status(order_id: str) -> str | None:
        order = MOCK_ORDERS.get(order_id.upper())
        return order.status if order else None
    
    @staticmethod
    def get_tracking_info(order_id: str) -> dict | None:
        order = MOCK_ORDERS.get(order_id.upper())
        if order and order.tracking_number:
            return {
                "tracking_number": order.tracking_number,
                "carrier": "FastShip",
                "status": order.status,
                "estimated_delivery": "2-3 business days"
            }
        return None
    
    @staticmethod
    def cancel_order(order_id: str) -> dict:
        order = MOCK_ORDERS.get(order_id.upper())
        if not order:
            return {"success": False, "message": "Order not found"}
        if order.status == "delivered":
            return {"success": False, "message": "Cannot cancel delivered orders"}
        if order.status == "shipped":
            return {"success": False, "message": "Order already shipped. Please initiate a return instead."}
        return {"success": True, "message": f"Order {order_id} has been cancelled"}


class ProductTools:
    """Tools for product-related operations."""
    
    def __init__(self):
        self._rag = None
        self._web_search = None
    
    @property
    def rag(self):
        if self._rag is None:
            try:
                from src.rag import ProductKnowledgeBase
                self._rag = ProductKnowledgeBase()
            except Exception:
                pass
        return self._rag
    
    @property
    def web_search(self):
        if self._web_search is None:
            try:
                from src.web_search import WebSearchTool
                self._web_search = WebSearchTool()
            except Exception:
                pass
        return self._web_search
    
    def search_products(self, query: str, category: str = None) -> list[dict]:
        results = []
        query_lower = query.lower()
        
        for product in MOCK_PRODUCTS.values():
            if query_lower in product.name.lower() or query_lower in product.description.lower():
                if category is None or product.category.lower() == category.lower():
                    results.append({
                        "product_id": product.product_id,
                        "name": product.name,
                        "price": product.price,
                        "in_stock": product.in_stock,
                        "category": product.category
                    })
        
        if self.rag and not results:
            try:
                rag_results = self.rag.search_products(query, category, limit=5)
                for r in rag_results:
                    if r.get("metadata"):
                        results.append({
                            "product_id": r["metadata"].get("product_id", ""),
                            "name": r["metadata"].get("name", ""),
                            "price": r["metadata"].get("price", 0),
                            "in_stock": r["metadata"].get("in_stock", False),
                            "category": r["metadata"].get("category", ""),
                            "relevance_score": r.get("score", 0),
                        })
            except Exception:
                pass
        
        return results
    
    def get_product_details(self, product_id: str) -> dict | None:
        product = MOCK_PRODUCTS.get(product_id.upper())
        if product:
            return {
                "product_id": product.product_id,
                "name": product.name,
                "price": product.price,
                "description": product.description,
                "category": product.category,
                "in_stock": product.in_stock,
                "specs": product.specs
            }
        return None
    
    def check_availability(self, product_id: str) -> dict | None:
        product = MOCK_PRODUCTS.get(product_id.upper())
        if product:
            return {
                "product_id": product.product_id,
                "name": product.name,
                "in_stock": product.in_stock,
                "message": "Available" if product.in_stock else "Currently out of stock"
            }
        return None
    
    def get_products_by_category(self, category: str) -> list[dict]:
        results = []
        for product in MOCK_PRODUCTS.values():
            if product.category.lower() == category.lower():
                results.append({
                    "product_id": product.product_id,
                    "name": product.name,
                    "price": product.price,
                    "in_stock": product.in_stock
                })
        return results
    
    def search_web_for_product(self, product_name: str) -> dict | None:
        if self.web_search and self.web_search.tavily.is_available():
            return self.web_search.search_product_info(product_name)
        return None


class SupportTools:
    """Tools for customer support operations."""
    
    def __init__(self):
        self._web_search = None
    
    @property
    def web_search(self):
        if self._web_search is None:
            try:
                from src.web_search import WebSearchTool
                self._web_search = WebSearchTool()
            except Exception:
                pass
        return self._web_search
    
    @staticmethod
    def get_faq(topic: str) -> dict | None:
        faqs = {
            "return": {
                "topic": "Returns & Refunds",
                "content": "You can return most items within 30 days of delivery for a full refund. "
                          "Items must be unused and in original packaging."
            },
            "shipping": {
                "topic": "Shipping Information",
                "content": "We offer standard shipping (5-7 business days) and express shipping (2-3 business days). "
                          "Free shipping on orders over 200 TL."
            },
            "warranty": {
                "topic": "Warranty Information",
                "content": "Most electronics come with a 2-year manufacturer warranty. "
                          "Accessories have a 1-year warranty."
            },
            "payment": {
                "topic": "Payment Methods",
                "content": "We accept credit cards (Visa, Mastercard), debit cards, and bank transfers. "
                          "Installment options available for orders over 500 TL."
            }
        }
        for key, faq in faqs.items():
            if key in topic.lower():
                return faq
        return None
    
    @staticmethod
    def create_support_ticket(subject: str, description: str) -> dict:
        import random
        ticket_id = f"TKT-{random.randint(10000, 99999)}"
        return {
            "ticket_id": ticket_id,
            "subject": subject,
            "status": "open",
            "message": f"Support ticket {ticket_id} created. Our team will respond within 24 hours."
        }
    
    @staticmethod
    def get_contact_info() -> dict:
        return {
            "phone": "+90 212 555 0123",
            "email": "support@techstore.com",
            "hours": "Monday-Friday 9:00-18:00, Saturday 10:00-14:00",
            "live_chat": "Available on website during business hours"
        }
    
    def search_web_for_help(self, topic: str) -> dict | None:
        if self.web_search and self.web_search.tavily.is_available():
            return self.web_search.search_support_info(topic)
        return None
