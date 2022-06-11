import React from "react";
import { Button, Card, Container } from "react-bootstrap";

function testForm() {
    return (
        <Container className="h-100">
            
            {/* main screen */}
            <Card className="my-auto">
                <Card.Body>
                    <div className="d-grid gap-2">
                        <h1> test form </h1> 
                    </div>
                </Card.Body>
            </Card>

            {/* endregion */}

        </Container>
    );
}

export default testForm;